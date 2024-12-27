#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, patch_norm_mse_loss, patch_norm_mse_loss_global
from utils.graphics_utils import patch_offsets, patch_warp, depth_propagation_1, check_geometric_consistency
import sys
from gaussian_renderer import render, network_gui, render_for_depth
import sys, time
from scene import Scene_allload, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)
from scene.app_model import AppModel
from scene.cameras import Camera
import multiprocessing as mp
from multiprocessing import Process 
from utils.camera_utils import cameraList_from_camInfos
from torchmetrics.functional.regression import pearson_corrcoef
from depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F





def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])




def training(dataset, opt, pipe,  testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene_allload(dataset, gaussians,partiton=False)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)
    propagation_dict = {}
    for i in range(0, len(scene.train_cameras_list), 1):
        propagation_dict[scene.train_cameras_list[i].image_name] = False
    patch_range = (5, 17)
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        randidx = randint(0, len(scene.train_cameras_list)-1)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack[randidx]
        # viewpoint_cam = cameraList_from_camInfos(scene.train_cameras_list[randidx:randidx+1], 1.0, dataset)[0]
        intervals = [-5, -3, 3, 5]
        src_idxs = [randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < len(scene.train_cameras_list)))]


        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # depth_gt = model_dpt.infer_image(viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy())

        # depth_gt = torch.tensor(depth_gt,device="cuda")
        # depth_gt = 1/depth_gt
        # midas_depth = depth_gt.reshape(-1, 1)
        # depth_gt = midas_depth
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
        # if iteration > 0:
        #     loss_hard = 0
        #     render_pkg = render_for_depth(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
        #                     return_plane=True, return_depth_normal=True)
        #     depth = render_pkg['plane_depth']        loss_hard = 0
        #     loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_gt[None,...].unsqueeze(0), randint(patch_range[0], patch_range[1]), opt.error_tolerance)
        #     loss_hard += 0.1 * loss_l2_dpt

        #     loss_global = patch_norm_mse_loss_global(depth[None,...], depth_gt[None,...].unsqueeze(0), randint(patch_range[0], patch_range[1]), opt.error_tolerance)
        #     loss_hard += 1 * loss_global
        #     loss_hard.backward()
        #     gaussians.optimizer.step()
        #     gaussians.optimizer.zero_grad(set_to_none = True)
        # surf_depth = render_pkg['plane_depth']

                # Depth regularization
 
            # Ll1depth = Ll1depth.item()


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=True, return_depth_normal=True)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        



        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        


        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg['plane_depth']
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            # depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) ).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth



        rendered_distance = render_pkg['rendered_distance'].reshape(-1, 1)

        rendered_distance = (rendered_distance - rendered_distance.min()) / (rendered_distance.max() - rendered_distance.min() + 1e-20)
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss



        if iteration > opt.single_view_weight_from_iter:



            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                # image_weight = erode(image_weight[None,None]).squeeze()
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += (normal_loss)


        if iteration > 0:
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            render_normal = (normal.permute(1,2,0) @ (viewpoint_cam.world_view_transform[:3,:3].T)).permute(2,0,1)
            rendered_depth_normal = (depth_normal.permute(1,2,0) @ (viewpoint_cam.world_view_transform[:3,:3].T)).permute(2,0,1)

        # if lambda_normal_prior > 0 and viewpoint_cam.normal_prior is not None:
            prior_normal = viewpoint_cam.noraml_gt * (render_pkg["rendered_alpha"]).detach()  #不透明的点影响越大 透明度大的点 较小
            prior_normal_mask = viewpoint_cam.normal_mask[0]

            normal_prior_error = (1 - F.cosine_similarity(prior_normal, render_normal, dim=0)) + \
                                (1 - F.cosine_similarity(prior_normal, rendered_depth_normal, dim=0))
            normal_prior_error = normal_prior_error  
            normal_prior_error = ranking_loss(normal_prior_error[prior_normal_mask], 
                                            penalize_ratio=1.0, type='mean')
            
            normal_prior_loss = 0.08 * normal_prior_error
            loss += (normal_prior_loss)



        # viewpoint_cam.nearest_id = scene.train_cameras_list[randidx].nearest_id 
        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            # index_near = random.sample(viewpoint_cam.nearest_id,1)[0] 
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            # loss += ncc_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            # Densification
        propagated_iteration_begin = opt.propagated_iteration_begin
        propagated_iteration_after = opt.propagated_iteration_after



        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0 and not propagation_dict[viewpoint_cam.image_name]):
                # if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):
                    propagation_dict[viewpoint_cam.image_name] = True

                    # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                    #             return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=True, return_depth_normal=True)

                    projected_depth = render_pkg["plane_depth"]

                    # get the opacity that less than the threshold, propagate depth in these region
                    # if viewpoint_cam.sky_mask is not None:
                    #     sky_mask = viewpoint_cam.sky_mask.to(opacity_mask.device).to(torch.bool)
                    # else:
                    #     sky_mask = None
                    # torchvision.utils.save_image(viewpoint_cam.original_image, "cost/"+viewpoint_cam.image_name+"_"+str(iteration)+"gt.png")

                    # get the propagated depth
                    propagated_depth, normal = depth_propagation_1(viewpoint_cam, projected_depth, scene.getTrainCameras(), src_idxs, opt.patch_size, dataset)
                    # cache the propagated_depth
                    viewpoint_cam.depth = propagated_depth

                    #transform normal to camera coordinate
                    R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                    # R_w2c[:, 1:] *= -1
                    normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                    valid_mask = propagated_depth != 300

                    # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                    plane_depth = render_pkg['plane_depth']
                    abs_rel_error = torch.abs(propagated_depth - plane_depth) / propagated_depth
                    abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - propagated_iteration_begin) / (propagated_iteration_after - propagated_iteration_begin)
                    # color error
                    render_color = render_pkg['render']
                    # torchvision.utils.save_image(render_color, "cost/"+viewpoint_cam.image_name+"_"+str(iteration)+"color.png")

                    color_error = torch.abs(render_color - viewpoint_cam.original_image)
                    color_error = color_error.mean(dim=0).squeeze()
                    error_mask = (abs_rel_error > abs_rel_error_threshold)

                    # # calculate the photometric consistency
                    ref_K = viewpoint_cam.K
                    #c2w
                    ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                                        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                    #             return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                    # calculate the geometric consistency
                    geometric_counts = None
                    for idx, src_idx in enumerate(src_idxs):
                        src_viewpoint = scene.getTrainCameras()[src_idx]
                        #c2w
                        src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                        src_K = src_viewpoint.K

                        if src_viewpoint.depth is None:
                            # src_render_pkg = render(src_viewpoint, gaussians, pipe, bg, 
                            #         return_normal=opt.normal_loss, return_opacity=False, return_depth=opt.depth_loss or opt.depth2normal_loss)
                            src_render_pkg  = render(src_viewpoint, gaussians, pipe, bg, app_model=app_model,
                                return_plane=True, return_depth_normal=True)
                            src_projected_depth = src_render_pkg['plane_depth']
                        
                        #get the src_depth first
                            src_depth, src_normal = depth_propagation_1(src_viewpoint, src_projected_depth, scene.getTrainCameras(), src_idxs, opt.patch_size,dataset)
                            src_viewpoint.depth = src_depth
                        else:
                            src_depth = src_viewpoint.depth
                            
                        mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                    ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                    src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                        
                        if geometric_counts is None:
                            geometric_counts = mask.to(torch.uint8)
                        else:
                            geometric_counts += mask.to(torch.uint8)
                            
                    cost = geometric_counts.squeeze()
                    cost_mask = cost >= 2       
                    
                    normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                    viewpoint_cam.normal = normal
                    
                    propagated_mask = valid_mask & error_mask & cost_mask
                    # if sky_mask is not None:
                    #     propagated_mask = propagated_mask & sky_mask

                    propagated_depth[~cost_mask] = 300 
                    # propagated_mask = propagated_mask & edge_mask
                    propagated_mask = propagated_mask.squeeze(0)
                    propagated_depth[~propagated_mask] = 300
    
                    if propagated_mask.sum() > 100:
                        gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image) 

            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene_allload, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def parallel_local_training(gpu_id, client_index, lp_args, op_args, pp_args,  test_iterations, save_iterations, checkpoint_iterations,
                            start_checkpoint, debug_from):
    torch.cuda.set_device(gpu_id)
    
    model_path = lp_args.model_path
    client_model_path = f"{model_path}/{client_index:05d}"
    lp_args.model_path = client_model_path

    training(lp_args, op_args, pp_args, test_iterations, save_iterations, checkpoint_iterations,start_checkpoint, debug_from)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene_allload, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--clients",type=int, default = 4)
    # parser.add_argument("--block_json_path",type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    mp.set_start_method('spawn', force=True)
    cuda_devices = torch.cuda.device_count()
    print(f'Found {cuda_devices} CUDA devices')
    trainin_round = args.clients // cuda_devices

    # model_dpt = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    # model_dpt.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cuda'))
    # model_dpt.to('cuda')
    # model_dpt.eval()


    for i in range(trainin_round):
        client_pool = [i+ trainin_round*j for j in range(cuda_devices)]
        process = []
        for index, device_id in enumerate(range(cuda_devices)):
            # if i < 4 :
            #     continue
            client_index = client_pool[index]
            p = Process(target=parallel_local_training, name=f"Client_{client_index}",
                        args=(device_id, client_index, lp.extract(args), op.extract(args), pp.extract(args), 
                              args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from))
            process.append(p)
            p.start()
        for p in process:
            p.join()
            process=[]
        torch.cuda.empty_cache()


    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # All done
    print("\nTraining complete.")
