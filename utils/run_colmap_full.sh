#!/bin/bash

set -e  # Exit immediately if any command exits with a non-zero status

# Function to log an error message and exit the script
log_and_exit() {
  echo "[Error] $1"
  exit 1
}

# Define paths for images, database, and reconstruction outputs
IMAGE_PATH="./color"
DATABASE_PATH="./database.db"
SPARSE_DIR="./sparse"
DENSE_DIR="./dense"
DEPTH_HINT_DIR="./depth"  # (Note: Not used in this script but defined)

echo "1. Feature extraction..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" || log_and_exit "Feature extraction failed"

echo "2. Feature matching..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH" || log_and_exit "Feature matching failed"

echo "3. Sparse reconstruction..."
mkdir -p "$SPARSE_DIR"
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_DIR" || log_and_exit "Sparse reconstruction failed"

echo "4. Image undistortion..."
mkdir -p "$DENSE_DIR"
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size 2000 || log_and_exit "Image undistortion failed"

echo "5. Dense matching using existing depth maps (PatchMatch Stereo)..."
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.max_image_size 2000 || log_and_exit "Dense matching failed"

echo "6. Dense point cloud fusion..."
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply" || log_and_exit "Dense point cloud fusion failed"

echo "7. Poisson mesh reconstruction..."
colmap poisson_mesher \
    --input_path "$DENSE_DIR/fused.ply" \
    --output_path "$DENSE_DIR/meshed-poisson.ply" || log_and_exit "Poisson mesh reconstruction failed"

echo "8. Delaunay mesh reconstruction..."
colmap delaunay_mesher \
    --input_path "$DENSE_DIR" \
    --input_type dense \
    --output_path "$DENSE_DIR/meshed-delaunay.ply" || log_and_exit "Delaunay mesh reconstruction failed"

echo "All steps completed successfully!"
