import pyrealsense2 as rs
import numpy as np
import cv2
import os

# -----------------------------
# 配置 RealSense 管道
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # 深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB流

# 启动管道
profile = pipeline.start(config)

# 对齐设置（把深度对齐到RGB）
align_to = rs.stream.color
align = rs.align(align_to)

# 创建保存路径
save_path = "data"
os.makedirs(save_path, exist_ok=True)
frame_id = 0

print("按 's' 保存RGB+Depth，按 'q' 退出")

try:
    while True:
        # 等待一帧
        frames = pipeline.wait_for_frames()

        # 对齐
        aligned_frames = align.process(frames)
        aligned_depth = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth or not color_frame:
            continue

        # 转 numpy 数组
        depth_image = np.asanyarray(aligned_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 伪彩色深度图（仅用于显示）
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 显示
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)

        # 键盘操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 保存RGB和深度
            cv2.imwrite(os.path.join(save_path, f"color_{frame_id:06d}.png"), color_image)
            cv2.imwrite(os.path.join(save_path, f"depth_{frame_id:06d}.png"), depth_image)  # 16位PNG
            print(f"保存第 {frame_id} 帧")
            frame_id += 1

        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
