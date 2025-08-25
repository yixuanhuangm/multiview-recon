import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time


def get_save_dir(base_dir="realsense_capture"):
    if not os.path.exists(base_dir):
        return base_dir
    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            return new_dir
        i += 1


def save_image(save_dir, image_index, color_image, depth_image):
    os.makedirs(os.path.join(save_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)

    color_path = os.path.join(save_dir, "color", f"color_{image_index:03d}.png")
    depth_path = os.path.join(save_dir, "depth", f"depth_{image_index:03d}.png")

    cv2.imwrite(color_path, color_image)
    cv2.imwrite(depth_path, depth_image)

    print(f"[{image_index}] Saved color image to {color_path}")
    print(f"[{image_index}] Saved depth image to {depth_path}")

def save_rgbd_with_mask(save_dir, image_index, color_image, depth_image, depth_threshold=800):
    # --- 确保文件夹 ---
    raw_color_dir = os.path.join(save_dir, "raw_color")
    raw_depth_dir = os.path.join(save_dir, "raw_depth")
    mask_color_dir = os.path.join(save_dir, "mask_color")
    mask_depth_dir = os.path.join(save_dir, "mask_depth")
    os.makedirs(raw_color_dir, exist_ok=True)
    os.makedirs(raw_depth_dir, exist_ok=True)
    os.makedirs(mask_color_dir, exist_ok=True)
    os.makedirs(mask_depth_dir, exist_ok=True)

    # --- 保存原始 ---
    raw_color_path = os.path.join(raw_color_dir, f"color_{image_index:03d}.png")
    raw_depth_path = os.path.join(raw_depth_dir, f"depth_{image_index:03d}.png")
    cv2.imwrite(raw_color_path, color_image)
    cv2.imwrite(raw_depth_path, depth_image)

    # --- 深度过滤 ---
    depth_mask = (depth_image > -depth_threshold) & (depth_image < depth_threshold)
    depth_mask = depth_mask.astype(np.uint8) * 255

    # --- 颜色过滤 ---
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    color_mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- 合并 mask ---
    combined_mask = cv2.bitwise_and(depth_mask, color_mask)

    # --- 过滤后的图像 ---
    color_masked = cv2.bitwise_and(color_image, color_image, mask=combined_mask)
    depth_masked = cv2.bitwise_and(depth_image, depth_image, mask=combined_mask)

    # --- 保存过滤后 ---
    mask_color_path = os.path.join(mask_color_dir, f"color_{image_index:03d}.png")
    mask_depth_path = os.path.join(mask_depth_dir, f"depth_{image_index:03d}.png")
    cv2.imwrite(mask_color_path, color_masked)
    cv2.imwrite(mask_depth_path, depth_masked)

    print(f"[{image_index}] Raw saved to {raw_color_path}, {raw_depth_path}")
    print(f"[{image_index}] Masked saved to {mask_color_path}, {mask_depth_path}")



def capture_rgbd_images(full_save_dir, num_images=1000, interval=2.0, capture_mode="automatic"):
    # --- Initialize RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # --- Start streaming ---
    profile = pipeline.start(config)

    # --- Align depth to color ---
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Store previous captured RGB and Depth (start as black images)
    previous_color = np.zeros((720, 1280, 3), dtype=np.uint8)
    previous_depth_colormap = np.zeros((720, 1280, 3), dtype=np.uint8)

    max_display_width = 1680
    image_index = 0
    last_capture_timestamp = 0

    try:
        while image_index < num_images:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Depth colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # --- Combine current RGB+Depth vertically ---
            current_pair = np.vstack((color_image, depth_colormap))

            # --- Combine previous RGB+Depth vertically ---
            previous_pair = np.vstack((previous_color, previous_depth_colormap))

            # --- Final preview: Left = current, Right = previous ---
            preview_frame = np.hstack((current_pair, previous_pair))

            # Resize if too wide
            if preview_frame.shape[1] > max_display_width:
                scale = max_display_width / preview_frame.shape[1]
                resized_preview = cv2.resize(preview_frame, (0, 0), fx=scale, fy=scale)
            else:
                resized_preview = preview_frame

            cv2.imshow("Realsense Capture (Top: RGB | Bottom: Depth)", resized_preview)
            key = cv2.waitKey(1) & 0xFF

            # --- Handle capture mode ---
            if capture_mode == "manual":
                if key == ord(' '):  # SPACE to save
                    save_rgbd_with_mask(full_save_dir, image_index, color_image, depth_image)
                    previous_color = color_image.copy()
                    previous_depth_colormap = depth_colormap.copy()
                    image_index += 1
            else:  # automatic
                current_timestamp = time.time()
                if current_timestamp - last_capture_timestamp >= interval:
                    save_rgbd_with_mask(full_save_dir, image_index, color_image, depth_image)
                    last_capture_timestamp = current_timestamp
                    previous_color = color_image.copy()
                    previous_depth_colormap = depth_colormap.copy()
                    image_index += 1

            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture RGB-D images from RealSense camera.")
    parser.add_argument("save_dir", nargs="?", default="realsense_capture", help="Directory to save images")
    parser.add_argument("num_images", nargs="?", type=int, default=1000, help="Number of images to capture")
    parser.add_argument("-i", "--interval", type=float, default=2.0, help="Interval between captures in seconds")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-m", "--manual", action="store_const", const="manual", dest="capture_mode",
                            help="Manual mode: press SPACE to capture")
    mode_group.add_argument("-a", "--automatic", action="store_const", const="automatic", dest="capture_mode",
                            help="Automatic mode: capture at fixed intervals")

    args = parser.parse_args()
    dataset_save_path = get_save_dir(args.save_dir)

    print(f"Saving data to folder: {dataset_save_path}")
    print(f"Number of images: {args.num_images}")
    print(f"Mode: {args.capture_mode}")
    if args.capture_mode == "automatic":
        print(f"Interval: {args.interval} seconds")

    capture_rgbd_images(
        os.path.join("../datasets", dataset_save_path),
        args.num_images,
        args.interval,
        args.capture_mode
    )
