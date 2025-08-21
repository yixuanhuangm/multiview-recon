# download_hf_subset.py

import argparse
from huggingface_hub import snapshot_download
import os


def download_subset(repo_id: str, subset_folder: str, local_dir: str = "./data"):
    """
    Download a specific subset (folder) from a Hugging Face dataset repo.

    Parameters
    ----------
    repo_id : str
        Hugging Face repo id, e.g. "username/my_dataset".
    subset_folder : str
        Folder name inside the repo you want to download, e.g. "human".
    local_dir : str
        Local path to save the downloaded data.
    """
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=f"{subset_folder}/*"   # 只下载 subset_folder/ 下的文件
    )

    print(f"✅ Subset '{subset_folder}' downloaded to {os.path.join(local_dir, subset_folder)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a specific subset from a Hugging Face dataset repo.")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Hugging Face dataset repo id, e.g. 'username/my_dataset'")
    parser.add_argument("--subset", type=str, required=True,
                        help="Subset folder name to download, e.g. 'human'")
    parser.add_argument("--local_dir", type=str, default="./data",
                        help="Local directory to save downloaded files (default: ./data)")

    args = parser.parse_args()

    download_subset(args.repo_id, args.subset, args.local_dir)
