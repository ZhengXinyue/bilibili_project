import time

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import cv2

import open3d as o3d
import open3d.core as o3c

from camera_lidar_fusion.fusion import read_bin, read_calib, get_colored_depth


def o3d_points_to_depth(points, height, width, intrinsic, extrinsic):
    """
    三维点云投影生成深度图
    @param points:       np.ndarray  [N, 3]
    @param height:       int
    @param width:        int
    @param intrinsic:    np.ndarray  [3, 3]
    @param extrinsic:    np.ndarray  [4, 4]
    @return:             np.ndarray  [H, W]    float32
    """
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, dtype=o3c.float32))
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_depth_image = pcd.project_to_depth_image(width=width, height=height,
                                                 intrinsics=intrinsic, extrinsics=extrinsic,
                                                 depth_scale=1, depth_max=100)
    return np.asarray(o3d_depth_image).squeeze()


def o3d_depth_to_points(depth, intrinsic, extrinsic):
    """
    深度图反投影生成点云
    @param depth:         np.ndarray  [H, W]   float32
    @param intrinsic:     np.ndarray  [3, 3]
    @param extrinsic:     np.ndarray  [4, 4]
    @return:              np.ndarray  [N, 3]
    """
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_depth_image = o3d.t.geometry.Image(o3c.Tensor(depth, dtype=o3c.float32))
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth=o3d_depth_image,
                                                            intrinsics=intrinsic, extrinsics=extrinsic,
                                                            depth_scale=1, depth_max=100)
    positions = pcd.point['positions']
    return positions.numpy()


def o3d_rgbd_to_points(rgb, depth, intrinsic, extrinsic):
    """
    rgbd图像反投影生成点云
    @param rgb:           np.ndarray  [H, W, 3]  uint8   RGB channel index
    @param depth:         np.ndarray  [H, W]     float32
    @param intrinsic:     np.ndarray  [3, 3]
    @param extrinsic:     np.ndarray  [4, 4]
    @return:
        positions:        np.ndarray  [N, 3]
        colors:           np.ndarray  [N, 3]   [0.0, 1.0]
    """
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    rgbd_image = o3d.t.geometry.RGBDImage(
        color=o3d.t.geometry.Image(o3c.Tensor(rgb, dtype=o3c.uint8)),
        depth=o3d.t.geometry.Image(o3c.Tensor(depth, dtype=o3c.float32)),
        aligned=True)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                           intrinsics=intrinsic, extrinsics=extrinsic,
                                                           depth_scale=1, depth_max=100)
    positions = pcd.point['positions']
    colors = pcd.point['colors']
    return positions.numpy(), colors.numpy()


def o3d_points_to_rgbd(points, colors, height, width, intrinsic, extrinsic):
    """
    着色点云投影生成rgbd图像
    @param points:       np.ndarray  [N, 3]
    @param colors:       np.ndarray  [N, 3]
    @param height:       int
    @param width:        int
    @param intrinsic:    np.ndarray  [3, 3]
    @param extrinsic:    np.ndarray  [4, 4]
    @return:
        color_image      np.ndarray  [H, W, 3]  uint8   RGB channel index
        depth_image      np.ndarray  [H, W]     float32
    """
    pcd = o3d.t.geometry.PointCloud()
    pcd.point['positions'] = o3c.Tensor(points, dtype=o3c.float32)
    pcd.point['colors'] = o3c.Tensor(colors, dtype=o3c.float32)
    intrinsic = o3c.Tensor(intrinsic, dtype=o3c.float32)
    extrinsic = o3c.Tensor(extrinsic, dtype=o3c.float32)

    o3d_rgbd_image = pcd.project_to_rgbd_image(width=width, height=height,
                                               intrinsics=intrinsic, extrinsics=extrinsic,
                                               depth_scale=1, depth_max=100)
    color_image = np.asarray(np.asarray(o3d_rgbd_image.color) * 255, dtype=np.uint8)
    depth_image = np.asarray(o3d_rgbd_image.depth).squeeze()
    return color_image, depth_image


if __name__ == '__main__':
    # my_image: 使用我的方法生成的深度图
    # o3d_image: 使用open3d中的方法生成的深度图
    image_path = '../data_example/3d_detection/image_2/000007.png'
    bin_path = '../data_example/3d_detection/velodyne/000007.bin'
    calib_path = '../data_example/3d_detection/calib/000007.txt'
    point_in_lidar = read_bin(bin_path)
    color_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    _, _, P2, _, R0, lidar2camera_matrix, _ = read_calib(calib_path)
    intrinsic = P2[:, :3]  # 内参
    extrinsic = np.matmul(R0, lidar2camera_matrix)  # 雷达到相机外参
    height, width = color_image.shape[:2]  # 图像高和宽

    depth1 = o3d_points_to_depth(point_in_lidar, height, width, intrinsic, extrinsic)
    # new_points = o3d_depth_to_points(depth, intrinsic, extrinsic)

    # colored_depth = get_colored_depth(depth1)
    # cv2.imshow('colored_depth', colored_depth)
    # cv2.waitKey()

    new_points, colors = o3d_rgbd_to_points(color_image, depth1, intrinsic, extrinsic)
    color_image, depth2 = o3d_points_to_rgbd(new_points, colors, height, width, intrinsic, extrinsic)
    colored_depth = get_colored_depth(depth2)

    # 雷达点云显示
    app = pg.mkQApp('main')
    widget = gl.GLViewWidget()
    point_size = np.zeros(new_points.shape[0], dtype=np.float16) + 0.1

    points_item1 = gl.GLScatterPlotItem(pos=new_points, size=point_size, color=colors, pxMode=False)
    widget.addItem(points_item1)
    widget.show()
    pg.exec()
