#!/bin/bash
set -e

log_and_exit() {
  echo "[Error] $1"
  exit 1
}

IMAGE_PATH="./mask_color"
DATABASE_PATH="./database.db"
SPARSE_DIR="./sparse"
DENSE_DIR="./dense"

echo "1. Feature extraction..."
colmap feature_extractor \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_PATH" \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1 \
  --SiftExtraction.max_num_features 16384 || log_and_exit "Feature extraction failed"

echo "2. Feature matching..."
colmap exhaustive_matcher \
  --database_path "$DATABASE_PATH" \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.cross_check 1 \
  --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.max_error 4 || log_and_exit "Feature matching failed"

echo "3. Sparse reconstruction..."
mkdir -p "$SPARSE_DIR"
colmap mapper \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_PATH" \
  --output_path "$SPARSE_DIR" \
  --Mapper.multiple_models 1 \
  --Mapper.init_min_tri_angle 2 \
  --Mapper.min_num_matches 10 || log_and_exit "Sparse reconstruction failed"

echo "3.5 Selecting best sparse model..."
BEST_MODEL=""
BEST_COUNT=0
for MODEL_DIR in "$SPARSE_DIR"/*; do
  if [[ -d "$MODEL_DIR" ]]; then
    STATS=`colmap model_analyzer --path "$MODEL_DIR"  2>&1`

    IMG_COUNT=$(echo "$STATS" | grep "Registered images" | sed -E 's/.*Registered images: ([0-9]+)/\1/')
    PNT_COUNT=$(echo "$STATS" | grep "Points:" | sed -E 's/.*Points: ([0-9]+)/\1/')
    echo "  Model: $MODEL_DIR | Images: $IMG_COUNT | Points: $PNT_COUNT"
    if (( IMG_COUNT > BEST_COUNT )); then
      BEST_COUNT=$IMG_COUNT
      BEST_MODEL=$MODEL_DIR
    fi
  fi
done

echo "$BEST_MODEL"

if [[ -z "$BEST_MODEL" ]]; then
  log_and_exit "No valid sparse model found!"
fi

echo "✅ Selected best model: $BEST_MODEL with $BEST_COUNT registered images."

echo "4. Undistortion..."
mkdir -p "$DENSE_DIR"
colmap image_undistorter \
  --image_path "$IMAGE_PATH" \
  --input_path "$BEST_MODEL" \
  --output_path "$DENSE_DIR" \
  --output_type COLMAP \
  --max_image_size 2000 || log_and_exit "Image undistortion failed"

echo "5. Dense matching (PatchMatch Stereo)..."
colmap patch_match_stereo \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size 2000 || log_and_exit "Dense matching failed"

echo "6. Point cloud fusion..."
colmap stereo_fusion \
  --workspace_path "$DENSE_DIR" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE_DIR/fused.ply" || log_and_exit "Dense point cloud fusion failed"

echo "7. Poisson mesh reconstruction..."
colmap poisson_mesher \
  --input_path "$DENSE_DIR/fused.ply" \
  --output_path "$DENSE_DIR/meshed-poisson.ply" || log_and_exit "Poisson mesh failed"

echo "8. Delaunay mesh reconstruction..."
colmap delaunay_mesher \
  --input_path "$DENSE_DIR" \
  --input_type dense \
  --output_path "$DENSE_DIR/meshed-delaunay.ply" || log_and_exit "Delaunay mesh failed"

echo "🎉 All steps completed!"
