
import torch
from scene import Scene, Scene_test
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
from utils.camera_utils import cameraList_from_camInfos

import cv2
import torch

from depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import sys
from PIL import Image



def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def get_sorted_images(folder_path):
    # 获取所有.jpg文件的路径
    images_path = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")]
    images_name = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    # 按文件名排序（假设文件名格式为000001.jpg, 000002.jpg, ...）
    # images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    return images_path, images_name

def render_set(model_path, name, dataset, images_path, images_name ,gaussians, pipeline,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):

    render_normal_path = os.path.join(model_path, name, "stablenormal", "renders_normal")

    makedirs(render_normal_path, exist_ok=True)
        # Create predictor instance
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    predictor = predictor.to("cuda")
# Apply the model to the image
# normal_image = predictor(input_image)

# Save or display the result
# normal_image.save("output/normal_map.png")



    # raw_img = cv2.imread('your/image/path')
    # depth = model.infer_image(raw_img) # HxW raw depth map


    
    depths_tsdf_fusion = []
    for idx, path in enumerate(tqdm(images_path, desc="Rendering progress")):
        
        # Apply the model to the image

        input_image = Image.open(path)
        normal_image = predictor(input_image)

        # Save or display the result
        normal_image.save(os.path.join(render_normal_path, images_name[idx]))

        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, num_cluster: int, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene_test(dataset, gaussians, shuffle=False, partiton=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        images_path, images_name = get_sorted_images(dataset.source_path)
        if not skip_train:
            render_set(dataset.model_path, "train", dataset, images_path, images_name, gaussians, pipeline, 
                       max_depth=max_depth,use_depth_filter=use_depth_filter)



if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)
    args = parser.parse_args(sys.argv[1:])
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)