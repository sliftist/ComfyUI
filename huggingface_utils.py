import os
import sys
import logging
import subprocess
import time
import asyncio

# Cache for repository file lists
_repo_files_cache = {}


def get_repo_files(repo_id):
    if repo_id not in _repo_files_cache:
        logging.info("Fetching file list for repo: {}".format(repo_id))
        try:
            _repo_files_cache[repo_id] = list_huggingface_repo_files(repo_id)
        except Exception as e:
            logging.error("Failed to fetch repo files for {}: {}".format(repo_id, e))
            _repo_files_cache[repo_id] = []  # Cache empty list to avoid repeated failures
    else:
        logging.debug("Using cached file list for repo: {}".format(repo_id))
    
    return _repo_files_cache[repo_id]


def ensure_huggingface_hub():
    try:
        import huggingface_hub
        logging.info("huggingface_hub is already installed (version: {})".format(huggingface_hub.__version__))
    except ImportError:
        logging.info("huggingface_hub not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        logging.info("Successfully installed huggingface_hub")
        raise


def list_huggingface_repo_files(repo_id):
    from huggingface_hub import HfApi

    logging.info("Listing files in repository: {}".format(repo_id))

    try:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=repo_id,
            token=os.environ.get('HF_TOKEN')
        )
        logging.info("Found {} files in {}".format(len(files), repo_id))
        return files
    except Exception as e:
        logging.error("Failed to list repository files: {}".format(e))
        raise


async def download_huggingface_model(repo_id, filename):
    from huggingface_hub import hf_hub_download

    # Download to ComfyUI models directory
    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    logging.info("Downloading {} from {} to {}".format(filename, repo_id, local_base_dir))

    try:
        os.makedirs(local_base_dir, exist_ok=True)

        start_time = time.time()
        file_path = await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            local_dir=local_base_dir,
            token=os.environ.get('HF_TOKEN')
        )
        end_time = time.time()
        
        # Calculate download statistics
        download_time = end_time - start_time
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        download_rate_mbps = file_size_mb / download_time if download_time > 0 else 0
        
        logging.info("Successfully downloaded model to: {}".format(file_path))
        logging.info("Download size: {:.2f} MB".format(file_size_mb))
        logging.info("Download time: {:.2f} seconds".format(download_time))
        logging.info("Download rate: {:.2f} MB/s".format(download_rate_mbps))
        return file_path
    except Exception as e:
        logging.error("Failed to download model: {}".format(e))
        raise


async def download_models(obj, repo_id):
    # Recursively find all strings
    def find_all_strings(data, found=None):
        if found is None:
            found = set()

        if isinstance(data, dict):
            for value in data.values():
                find_all_strings(value, found)
        elif isinstance(data, list):
            for item in data:
                find_all_strings(item, found)
        elif isinstance(data, str):
            # Add any non-empty string
            if data.strip():
                found.add(data)

        return found

    all_strings = find_all_strings(obj)

    if not all_strings:
        logging.info("No strings found in object")
        return []

    logging.info("Found {} string(s) to check against repo".format(len(all_strings)))

    # Get cached repo files
    repo_files = get_repo_files(repo_id)

    # Local base directory for models
    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    downloaded_paths = []

    # Match each string against repo files
    for potential_filename in all_strings:
        # Find all matching files in repo (comparing just the basename)
        matches = [f for f in repo_files if os.path.basename(f) == potential_filename]

        if not matches:
            # Only warn if it looks like a safetensors file we expected to find
            if potential_filename.endswith('.safetensors'):
                logging.warning("No matches found in repo for safetensors file: {}".format(potential_filename))
            continue

        if len(matches) > 1:
            logging.warning("Multiple matches found for '{}': {}".format(potential_filename, matches))
            logging.warning("Using first match: {}".format(matches[0]))

        # Take the first match
        repo_file_path = matches[0]

        # Check if file exists locally
        local_file_path = os.path.join(local_base_dir, repo_file_path)

        if os.path.exists(local_file_path):
            logging.info("File already exists locally: {}".format(local_file_path))
            downloaded_paths.append(local_file_path)
        else:
            logging.info("File not found locally, downloading: {}".format(repo_file_path))
            try:
                downloaded_path = await download_huggingface_model(repo_id, repo_file_path)
                downloaded_paths.append(downloaded_path)
            except Exception as e:
                logging.error("Failed to download {}: {}".format(repo_file_path, e))

    return downloaded_paths

