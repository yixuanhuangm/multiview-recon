#!/bin/bash

set -e

IMAGE_PATH="./color"
DATABASE_PATH="./database.db"
SPARSE_DIR="./sparse"
DENSE_DIR="./dense"
DEPTH_HINT_DIR="./depth_hint"

echo "1. Feature extraction..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH"

echo "2. Feature matching..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "3. Sparse reconstruction..."
mkdir -p "$SPARSE_DIR"
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_DIR"

echo "4. Image undistortion..."
mkdir -p "$DENSE_DIR"
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$DENSE_DIR" \
    --output_type COLMAP \
    --max_image_size 2000

echo "5. Dense matching using depth hints..."
colmap patch_match_stereo \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true \
    --PatchMatchStereo.max_image_size 2000 \
    --PatchMatchStereo.depth_hint_path "$DEPTH_HINT_DIR" \
    --PatchMatchStereo.depth_hint_type depth_map

echo "6. Dense point cloud fusion..."
colmap stereo_fusion \
    --workspace_path "$DENSE_DIR" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE_DIR/fused.ply"

echo "7. Poisson mesh reconstruction..."
colmap poisson_mesher \
    --input_path "$DENSE_DIR/fused.ply" \
    --output_path "$DENSE_DIR/meshed-poisson.ply"

echo "8. Delaunay mesh reconstruction..."
colmap delaunay_mesher \
    --input_path "$DENSE_DIR" \
    --input_type dense \
    --output_path "$DENSE_DIR/meshed-delaunay.ply"

echo "All steps completed successfully!"
