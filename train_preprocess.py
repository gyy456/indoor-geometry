import os
from pathlib import Path

from colmap_utils.data_io import create_ply_from_colmap
from geometry.common import get_align_matrix

import numpy as np
import open3d as o3d

def fit_pointcloud_plane(ply_file):
    print(f'fit_pointcloud_plane function')
    print(f'reading file : {ply_file}')
    pcd = o3d.io.read_point_cloud(ply_file)
    # 拟合平面并计算法向量

    # 进行 Voxel 降采样
    voxel_size = 0.05  # 设置体素大小
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=10000)

    # 获取平面参数
    [a, b, c, d] = plane_model

    # 计算平面法向量
    plane_normal = [a, b, c]

    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # print(f"Plane normal: {plane_normal}")
    print(f"Plane inliers: {len(inliers)}")

    # 显示点云和拟合的平面
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 将平面点云颜色设为红色

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([coord_frame, inlier_cloud, outlier_cloud])

    print('\n')
    return [a, b, c, d]

def save_align_pcd(ply_file, align_matrix, save_aligned_pcd_path = None):
    print(f'checking out align matrix')
    print(f'reading file : {ply_file} ')
    print(f'align_matrix : {align_matrix}')
    
    pcd = o3d.io.read_point_cloud(ply_file)
    original_pcd = o3d.io.read_point_cloud(ply_file)
    aligned_pcd = pcd.transform(align_matrix)
    
    original_pcd.paint_uniform_color([0, 0, 0])
    aligned_pcd.paint_uniform_color([0, 1, 0])
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([coord_frame, original_pcd, aligned_pcd])

    if save_aligned_pcd_path is not None:
      print(f'saving aligned pcd in: {save_aligned_pcd_path}')
      o3d.io.write_point_cloud(save_aligned_pcd_path, aligned_pcd)

    print('\n')

def align_pointcloud(ply_file, train_data_path):
  plane_model = fit_pointcloud_plane(ply_file)
  plane_norm = [plane_model[0], plane_model[1], plane_model[2]]
  align_matrix = get_align_matrix(vec1=plane_norm, vec2=np.array([0, 0, -1]))
  align_matrix_path = train_data_path + '/align_matrix.txt'
  np.savetxt(align_matrix_path, align_matrix)
  original_pcd = o3d.io.read_point_cloud(ply_file)
  aligned_pcd = original_pcd.transform(align_matrix)
  o3d.io.write_point_cloud(train_data_path + '/pointcloud/aligned_points3D.ply', aligned_pcd)

def get_3d_box(box_pcd, save_path):
    bboxs = np.array([])
    class_id = 40   # bottle is 40 in nyuid
    instance_id = 1
    pcd = o3d.io.read_point_cloud(box_pcd)
    aabb = pcd.get_axis_aligned_bounding_box()
    xyz = aabb.get_center()
    hwl = aabb.get_extent()
    bbox = np.concatenate([xyz, hwl, [class_id], [instance_id]])
    bboxs = np.concatenate([bboxs, bbox])
    bboxs = np.reshape(bboxs, [-1, 8])
    box_npy_name = save_path + "/bboxs_aabb.npy"
    np.save(box_npy_name, bboxs)

def checkout_3D_BOX(ply_file, align_matrix, bboxs):
    pcd = o3d.io.read_point_cloud(ply_file)
    aligned_pcd = pcd.transform(align_matrix)
    aligned_pcd.paint_uniform_color([1.0, 0, 0])

    pc_bbox = np.load(bboxs)
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = pc_bbox[0,0:3]
    bbox.extent = pc_bbox[0,3:6]
    bbox.color = (0,0,0)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([coord_frame, aligned_pcd, bbox])

def checkout_train_data(data_path):
    original_pcd_name = data_path + '/pointcloud/points3D.ply'
    aligned_pcd_name = data_path + '/pointcloud/aligned_points3D.ply'
    box_pcd_name = data_path + '/pointcloud/3D-BOX.ply'

    original_pcd = o3d.io.read_point_cloud(original_pcd_name)
    aligned_pcd = o3d.io.read_point_cloud(aligned_pcd_name)
    box_pcd = o3d.io.read_point_cloud(box_pcd_name)

    original_pcd.paint_uniform_color([0, 0, 0])
    aligned_pcd.paint_uniform_color([0, 1, 0])
    box_pcd.paint_uniform_color([1, 0, 0])

    bboxs = data_path + '/bboxs_aabb.npy'
    pc_bbox = np.load(bboxs)
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = pc_bbox[0,0:3]
    bbox.extent = pc_bbox[0,3:6]
    bbox.color = (0,0,1)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 5, origin=[1, 1, 0])
    o3d.visualization.draw_geometries([coord_frame, original_pcd, aligned_pcd, box_pcd, bbox])

def preprocess(train_data_path, coord_size = 0.1):
    colmap_ply_name = 'points3D.ply'
    recon_dir = train_data_path + '/sparse/0'
    pointcloud_dir = train_data_path + '/pointcloud'
    if not os.path.exists(pointcloud_dir):
        os.mkdir(pointcloud_dir)

    create_ply_from_colmap(colmap_ply_name, Path(recon_dir), Path(pointcloud_dir), None)

    colmap_ply_file = pointcloud_dir + '/' + colmap_ply_name
    align_pointcloud(colmap_ply_file, train_data_path)

    original_pcd_name = train_data_path + '/pointcloud/points3D.ply'
    aligned_pcd_name = train_data_path + '/pointcloud/aligned_points3D.ply'

    original_pcd = o3d.io.read_point_cloud(original_pcd_name)
    aligned_pcd = o3d.io.read_point_cloud(aligned_pcd_name)

    original_pcd.paint_uniform_color([0, 0, 0])
    aligned_pcd.paint_uniform_color([0, 1, 0])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(coord_size, origin=[1, 1, 0])
    o3d.visualization.draw_geometries([coord_frame, original_pcd, aligned_pcd])