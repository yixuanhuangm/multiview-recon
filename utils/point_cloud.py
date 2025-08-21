import open3d as o3d
import cv2
import numpy as np

# 路径设置
rgb_path = "color_002.png"
depth_path = "depth_002.png"

# 读取深度图
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 保留原始深度
height, width = depth.shape

# 读取 RGB 图像并缩放到深度图大小
color = cv2.imread(rgb_path)
color = cv2.resize(color, (width, height))
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

# 转换为 Open3D 图像
color_o3d = o3d.geometry.Image(color)
depth_o3d = o3d.geometry.Image(depth)

# 相机内参设置（根据深度图分辨率）
fx, fy = 525.0, 525.0  # 焦距，可根据相机调整
cx, cy = width / 2, height / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 创建 RGBD 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,
    depth_scale=1000.0,   # 如果深度图单位是毫米
    convert_rgb_to_intensity=False
)

# 从 RGBD 图像生成点云
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# 坐标系调整（可选）
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# 可视化点云
o3d.visualization.draw_geometries([pcd])

# 保存点云
o3d.io.write_point_cloud("output.ply", pcd)
print("点云已保存为 output.ply")
