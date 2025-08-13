import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time


def get_save_dir(base_dir="realsense_capture"):
    """
    Generate a unique directory name based on base_dir.
    If base_dir exists, append _1, _2, ... until an unused name is found.

    Args:
        base_dir (str): Base directory name.

    Returns:
        str: A unique directory name for saving images.
    """
    if not os.path.exists(base_dir):
        return base_dir
    else:
        i = 1
        while True:
            new_dir = f"{base_dir}_{i}"
            if not os.path.exists(new_dir):
                return new_dir
            i += 1


def capture_rgbd_images(save_dir, num_images=100, interval=1.0):
    """
    Capture RGB and Depth images from Intel RealSense camera.

    Args:
        save_dir (str): Directory where images will be saved.
        num_images (int): Number of images to capture.
        interval (float): Time interval (in seconds) between captures.
    """
    # Create directories for saving color and depth images
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)

    # Initialize RealSense pipeline and configuration
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color and depth streams at 640x480 resolution, 30 FPS
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)

    try:
        count = 0
        last_capture_time = 0
        while count < num_images:
            # Wait for a coherent pair of frames: color and depth
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue  # Skip if any frame is unavailable

            current_time = time.time()
            if current_time - last_capture_time >= interval:
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Generate file paths with zero-padded indices
                color_path = os.path.join(save_dir, "color", f"color_{count:03d}.png")
                depth_path = os.path.join(save_dir, "depth", f"depth_{count:03d}.png")

                # Save images to disk
                cv2.imwrite(color_path, color_image)
                cv2.imwrite(depth_path, depth_image)

                print(f"Saved color image to {color_path}")
                print(f"Saved depth image to {depth_path}")

                count += 1
                last_capture_time = current_time

    finally:
        # Stop the RealSense pipeline on exit
        pipeline.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture RGB-D images from RealSense camera."
    )
    parser.add_argument(
        "save_dir",
        nargs="?",
        default="realsense_capture",
        help="Directory to save images (default: realsense_capture)",
    )
    parser.add_argument(
        "num_images",
        nargs="?",
        type=int,
        default=100,
        help="Number of images to capture (default: 100)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Time interval between captures in seconds (default: 3.0)",
    )

    args = parser.parse_args()

    # Generate a unique save directory to avoid overwriting
    save_dir = get_save_dir(args.save_dir)
    print(f"Saving data to folder: {save_dir}")
    print(f"Number of images to capture: {args.num_images}")
    print(f"Capture interval: {args.interval} seconds")

    # Capture images and save to the specified directory
    capture_rgbd_images(os.path.join("../datasets", save_dir), args.num_images, args.interval)
