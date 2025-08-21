# Multiview Reconstruction

Contributed by Yixuan Huang

A lightweight toolkit for multi-view 3D reconstruction based on [COLMAP](https://colmap.github.io/).  
This repository mainly provides utility functions and a self-collected dataset captured with an **Intel RealSense D436**.

## Features

- Tools for preprocessing, visualization, and handling 3D reconstruction data.
- Self-collected dataset for testing and development.
- Demo scripts to quickly run and visualize reconstruction results.

## Dataset

- Captured using Intel RealSense D436.
- Consists of multiple images per scene for multi-view reconstruction.
- Organized for direct use with COLMAP pipelines.

## Requirements

- COLMAP installed and accessible in your environment.
- Python 3.10 for running utility scripts.
- Standard Python libraries: `numpy`, `opencv-python`, etc.

## Usage

1. Preprocess images and organize them in the dataset folder.  
2. Use provided utility scripts for feature extraction, matching, and visualization.  
3. Run COLMAP for reconstruction following standard pipelines.
