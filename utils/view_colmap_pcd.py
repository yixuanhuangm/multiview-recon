import open3d as o3d
import argparse
import os
import sys


def load_and_show_model(project_dir, model_type):
    """
    Load and visualize a dense reconstruction model (point cloud or mesh) from a COLMAP project.

    Args:
        project_dir (str): Root directory of the COLMAP project (expects a 'dense' subfolder).
        model_type (str): Type of model to load and display:
                          'f' - fused point cloud (.ply),
                          'p' - Poisson mesh (.ply),
                          'd' - Delaunay mesh (.ply).
    """
    dense_dir = os.path.join(project_dir, "dense")  # Path to 'dense' directory

    # Determine file path based on selected model type
    if model_type == "f":
        file_path = os.path.join(dense_dir, "fused.ply")
    elif model_type == "p":
        file_path = os.path.join(dense_dir, "meshed-poisson.ply")
    elif model_type == "d":
        file_path = os.path.join(dense_dir, "meshed-delaunay.ply")
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

    # Check if the model file exists
    if not os.path.isfile(file_path):
        print(f"File does not exist: {file_path}")
        sys.exit(2)

    print(f"Loading and displaying {model_type} model: {file_path}")

    # Load and visualize the model using Open3D
    if model_type == "f":
        # For fused point cloud, read as point cloud and display
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd])
    else:
        # For meshes, read as triangle mesh, compute normals, and display
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])


def main():
    parser = argparse.ArgumentParser(
        description="View COLMAP dense reconstruction models (point cloud or mesh)."
    )
    parser.add_argument(
        "project_dir",
        help="Root directory of the project; expects a 'dense' subfolder."
    )
    parser.add_argument(
        "-t", "--type",
        choices=["f", "p", "d"],
        default="f",
        help="Model type to view: f (Fused point cloud), p (Poisson mesh), d (Delaunay mesh). Default is 'f'."
    )
    args = parser.parse_args()

    # Prepend datasets path to the project directory
    full_project_dir = os.path.join("../datasets", args.project_dir)
    load_and_show_model(full_project_dir, args.type)


if __name__ == "__main__":
    main()
