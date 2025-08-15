import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time
import struct


def get_save_dir(base_dir="realsense_capture"):
    """
    Generate a unique directory name to avoid overwriting existing data.

    Args:
        base_dir (str): Base directory name.

    Returns:
        str: A unique directory name for saving images.
    """
    if not os.path.exists(base_dir):
        return base_dir

    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            return new_dir
        i += 1


def save_image(save_dir, image_index, color_image, depth_image, depth_hint):
    """
    Save RGB image, original depth image, and aligned depth hint in COLMAP BIN format.

    Args:
        save_dir (str): Base directory to save images.
        image_index (int): Current image index.
        color_image (ndarray): RGB image array.
        depth_image (ndarray): Original depth image (uint16, in mm).
        depth_hint (ndarray): Aligned depth in meters (float32).

    Outputs:
        - color PNG
        - depth PNG
        - depth_hint BIN (COLMAP compatible)
        Prints saved file paths.
    """
    # --- Generate file paths ---
    color_path = os.path.join(save_dir, "color", f"color_{image_index:03d}.png")
    depth_path = os.path.join(save_dir, "depth", f"depth_{image_index:03d}.png")
    depth_bin_path = os.path.join(save_dir, "depth_hint", f"color_{image_index:03d}.bin")

    # --- Write images to disk ---
    cv2.imwrite(color_path, color_image)
    cv2.imwrite(depth_path, depth_image)

    # --- Save depth_hint in COLMAP BIN format ---
    with open(depth_bin_path, "wb") as f:
        f.write(struct.pack('Q', depth_hint.shape[1]))  # width
        f.write(struct.pack('Q', depth_hint.shape[0]))  # height
        depth_hint.astype(np.float32).tofile(f)

    # --- Print saved file info ---
    print(f"[{image_index}] Saved color: {color_path}")
    print(f"[{image_index}] Saved depth (PNG): {depth_path}")
    print(f"[{image_index}] Saved depth_hint (BIN): {depth_bin_path}")


def capture_rgbd_images(full_save_dir, num_images=1000, interval=1.0, capture_mode="automatic"):
    """
    Capture RGB-D images from RealSense camera and save RGB, depth, and depth_hint.

    Args:
        full_save_dir (str): Directory to save images.
        num_images (int): Number of images to capture.
        interval (float): Time interval between captures (automatic mode).
        capture_mode (str): 'manual' or 'automatic'.

    Outputs:
        - Creates subfolders: color/, depth/, depth_hint/
        - Displays a preview window with live (left) and last captured (right)
        - Prints file paths when images are saved
    """

    # --- Create directories ---
    os.makedirs(os.path.join(full_save_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(full_save_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(full_save_dir, "depth_hint"), exist_ok=True)

    # --- Initialize RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    # --- Align depth to color ---
    align_to = rs.stream.color
    align = rs.align(align_to)

    previous_color_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    max_display_width = 1680
    image_index = 0
    last_capture_timestamp = 0

    try:
        pipeline.start(config)

        while image_index < num_images:
            # --- Capture frames ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # --- Convert to numpy arrays ---
            color_image = np.asanyarray(color_frame.get_data())
            depth_aligned = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # meters

            # --- Filter depth: set invalid or too far values to 0 ---
            depth_aligned[(depth_aligned <= 0) | (depth_aligned > 5.0)] = 0

            # --- Prepare preview window ---
            preview_frame = np.hstack((color_image, previous_color_image))
            if preview_frame.shape[1] > max_display_width:
                scale = max_display_width / preview_frame.shape[1]
                resized_preview = cv2.resize(preview_frame, (0, 0), fx=scale, fy=scale)
            else:
                resized_preview = preview_frame

            cv2.imshow("Realsense Capture (Left: Live | Right: Last Shot)", resized_preview)
            key = cv2.waitKey(1) & 0xFF

            # --- Save images ---
            if capture_mode == "manual":
                if key == ord(' '):  # Press SPACE to capture
                    save_image(
                        full_save_dir,
                        image_index,
                        color_image,
                        (depth_aligned * 1000).astype(np.uint16),  # convert meters to mm
                        depth_aligned
                    )
                    previous_color_image = color_image.copy()
                    image_index += 1
            else:
                current_timestamp = time.time()
                if current_timestamp - last_capture_timestamp >= interval:
                    save_image(
                        full_save_dir,
                        image_index,
                        color_image,
                        (depth_aligned * 1000).astype(np.uint16),
                        depth_aligned
                    )
                    last_capture_timestamp = current_timestamp
                    previous_color_image = color_image.copy()
                    image_index += 1

            # --- Quit ---
            if key == ord('q'):
                print("User requested exit. Stopping capture.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description="Capture RGB-D images from RealSense camera.")
    parser.add_argument("save_dir", nargs="?", default="realsense_capture")
    parser.add_argument("num_images", nargs="?", type=int, default=1000)
    parser.add_argument("-i", "--interval", type=float, default=3.0)

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-m", "--manual", action="store_const", const="manual", dest="capture_mode")
    mode_group.add_argument("-a", "--automatic", action="store_const", const="automatic", dest="capture_mode")

    args = parser.parse_args()

    # --- Generate unique save directory ---
    dataset_save_path = get_save_dir(args.save_dir)
    print(f"Saving to: {dataset_save_path}")
    print(f"Mode: {args.capture_mode}, Images: {args.num_images}, Interval: {args.interval}")

    # --- Start capture ---
    capture_rgbd_images(
        os.path.join("../datasets", dataset_save_path),
        args.num_images,
        args.interval,
        args.capture_mode
    )
