import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import time


def get_save_dir(base_dir="realsense_capture"):
    """
    Generate a unique directory name based on base_dir.
    If the base directory exists, append _1, _2, ... until an unused name is found.

    Args:
        base_dir (str): Base directory name.

    Returns:
        str: A unique directory name for saving images. This string can be used
             to create folders to store RGB-D images without overwriting existing data.
    """
    if not os.path.exists(base_dir):
        return base_dir

    i = 1
    while True:
        new_dir = f"{base_dir}_{i}"
        if not os.path.exists(new_dir):
            return new_dir
        i += 1


def save_image(save_dir, image_index, color_image, depth_image):
    """
    Save RGB and Depth images to disk with zero-padded filenames.

    Args:
        save_dir (str): Base directory to save images.
        image_index (int): Current image index.
        color_image (ndarray): RGB image array.
        depth_image (ndarray): Depth image array.

    Returns:
        None

    Side Effects:
        - Writes two image files to disk: one color image and one depth image.
        - Prints the saved file paths to the console.
    """
    # --- Generate file paths ---
    color_path = os.path.join(save_dir, "color", f"color_{image_index:03d}.png")
    depth_path = os.path.join(save_dir, "depth", f"depth_{image_index:03d}.png")

    # --- Write images to disk ---
    cv2.imwrite(color_path, color_image)
    cv2.imwrite(depth_path, depth_image)

    # --- Print saved file info ---
    print(f"[{image_index}] Saved color image to {color_path}")
    print(f"[{image_index}] Saved depth image to {depth_path}")


def capture_rgbd_images(full_save_dir, num_images=100, interval=1.0, capture_mode="automatic"):
    """
    Capture RGB-D images from Intel RealSense camera, supporting manual or automatic modes.

    Args:
        full_save_dir (str): Directory where images will be saved.
        num_images (int): Total number of images to capture.
        interval (float): Time interval (in seconds) between captures (only for automatic mode).
        capture_mode (str): "manual" or "automatic".

    Returns:
        None

    Side Effects:
        - Creates 'color' and 'depth' subdirectories under `full_save_dir`.
        - Saves captured RGB and Depth images to disk.
        - Displays a live preview window showing the current and last captured images.
        - Prints each saved file path to the console.
        - Stops the camera stream and closes OpenCV windows on exit.
    """

    # --- Create directories for saving ---
    os.makedirs(os.path.join(full_save_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(full_save_dir, "depth"), exist_ok=True)

    # --- Initialize RealSense pipeline ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    previous_color_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # --- Start streaming ---
    pipeline.start(config)
    max_display_width = 1680

    image_index = 0
    last_capture_timestamp = 0

    try:
        while image_index < num_images:

            # --- Capture frames ---
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # --- Prepare preview frame ---
            preview_frame = np.hstack((color_image, previous_color_image))
            if preview_frame.shape[1] > max_display_width:
                scale = max_display_width / preview_frame.shape[1]
                resized_preview = cv2.resize(preview_frame, (0, 0), fx=scale, fy=scale)
            else:
                resized_preview = preview_frame

            cv2.imshow("Realsense Capture (Left: Live | Right: Last Shot)", resized_preview)
            key = cv2.waitKey(1) & 0xFF

            # --- Handle capture mode ---
            if capture_mode == "manual":
                # Manual capture: press SPACE to save
                if key == ord(' '):
                    save_image(full_save_dir, image_index, color_image, depth_image)
                    previous_color_image = color_image.copy()
                    image_index += 1
            else:
                # Automatic capture at fixed intervals
                current_timestamp = time.time()
                if current_timestamp - last_capture_timestamp >= interval:
                    save_image(full_save_dir, image_index, color_image, depth_image)
                    last_capture_timestamp = current_timestamp
                    previous_color_image = color_image.copy()
                    image_index += 1

            # --- Quit condition ---
            if key == ord('q'):
                break

    finally:
        # --- Stop streaming and close windows ---
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Capture RGB-D images from RealSense camera.")
    parser.add_argument(
        "save_dir",
        nargs="?",
        default="realsense_capture",
        help="Directory to save images (default: realsense_capture)"
    )
    parser.add_argument(
        "num_images",
        nargs="?",
        type=int,
        default=100,
        help="Number of images to capture (default: 100)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=3.0,
        help="Time interval between captures in seconds (default: 3.0)"
    )

    # --- Mutually exclusive group: manual or automatic capture mode ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-m", "--manual",
        action="store_const",
        const="manual",
        dest="capture_mode",
        help="Enable manual capture mode: press SPACE to capture"
    )
    mode_group.add_argument(
        "-a", "--automatic",
        action="store_const",
        const="automatic",
        dest="capture_mode",
        help="Enable automatic capture mode: capture at fixed intervals"
    )

    args = parser.parse_args()

    # --- Generate a unique save directory ---
    dataset_save_path = get_save_dir(args.save_dir)

    print(f"Saving data to folder: {dataset_save_path}")
    print(f"Number of images to capture: {args.num_images}")
    print(f"Capture mode: {args.capture_mode}")
    if args.capture_mode == "automatic":
        print(f"Capture interval: {args.interval} seconds")

    # --- Start capture process ---
    capture_rgbd_images(
        os.path.join("../datasets", dataset_save_path),
        args.num_images,
        args.interval,
        args.capture_mode
    )
