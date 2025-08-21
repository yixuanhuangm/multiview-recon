import open3d as o3d
import numpy as np
import cv2

# 设置相机内参（根据你的相机调整）
width, height = 640, 480
fx, fy = 525.0, 525.0  # 焦距
cx, cy = width / 2, height / 2

intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# RGB 和 Depth 文件路径
rgb_path = "color_002.png"
depth_path = "depth_002.png"

# 读取图像
color = o3d.io.read_image(rgb_path)
depth = o3d.io.read_image(depth_path)

# 创建 RGBD 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, convert_rgb_to_intensity=False
)

# 从 RGBD 图像生成点云
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# 可选：调整坐标系
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# 可视化
o3d.visualization.draw_geometries([pcd])

# 保存点云
o3d.io.write_point_cloud("output.ply", pcd)
print("点云已保存为 output.ply")
