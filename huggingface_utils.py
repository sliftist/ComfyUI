import os
import sys
import logging
import subprocess
import time
import asyncio
import shutil

MINIMUM_FREE_SPACE_GB = 10

_repo_files_cache = {}
_repo_file_sizes_cache = {}
_file_access_times = {}


def initialize_access_times():
    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.exists(local_base_dir):
        return
    
    for root, dirs, files in os.walk(local_base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _file_access_times[file_path] = os.path.getatime(file_path)


def update_access_time(file_path):
    _file_access_times[file_path] = time.time()
    os.utime(file_path, None)


def delete_old_files_until_space(target_dir, required_bytes):
    total, used, free = shutil.disk_usage(target_dir)
    minimum_free = MINIMUM_FREE_SPACE_GB * 1024 * 1024 * 1024
    needed_space = required_bytes + minimum_free
    
    if free >= needed_space:
        logging.info("Need {} GB of space, but we have {} GB of free space. No files will be deleted.".format(needed_space / (1024 * 1024 * 1024), free / (1024 * 1024 * 1024)))
        return
    space_to_free = needed_space - free
    
    logging.info("Not enough free space. Deleting old files to free up space. Needed: {:.2f} GB, Available: {:.2f} GB (freeing {:.2f} GB)".format(needed_space / (1024 * 1024 * 1024), free / (1024 * 1024 * 1024), space_to_free / (1024 * 1024 * 1024)))
    
    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    files_with_times = []
    
    for file_path, access_time in _file_access_times.items():
        if os.path.exists(file_path) and file_path.startswith(local_base_dir):
            files_with_times.append((access_time, file_path))
    
    files_with_times.sort()
    
    freed_space = 0
    for access_time, file_path in files_with_times:
        if freed_space >= space_to_free:
            break
        
        file_size = os.path.getsize(file_path)
        os.remove(file_path)
        del _file_access_times[file_path]
        freed_space += file_size
        logging.info("Deleted {} ({:.2f} MB)".format(file_path, file_size / (1024 * 1024)))


def get_repo_files(repo_id):
    if repo_id not in _repo_files_cache:
        logging.info("Fetching file list for repo: {}".format(repo_id))
        _repo_files_cache[repo_id] = list_huggingface_repo_files(repo_id)
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

    api = HfApi()
    repo_info = api.model_info(repo_id=repo_id, files_metadata=True, token=os.environ.get('HF_TOKEN'))
    
    files_list = []
    sizes_dict = {}
    for sibling in repo_info.siblings:
        files_list.append(sibling.rfilename)
        sizes_dict[sibling.rfilename] = sibling.size
    
    _repo_file_sizes_cache[repo_id] = sizes_dict
    
    logging.info("Found {} files in {}".format(len(files_list), repo_id))
    return files_list


def get_file_size(repo_id, filename):
    if repo_id not in _repo_file_sizes_cache:
        get_repo_files(repo_id)
    return _repo_file_sizes_cache[repo_id].get(filename, 0)


async def download_huggingface_model(repo_id, filename):
    from huggingface_hub import hf_hub_download

    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(local_base_dir, exist_ok=True)
    
    file_size_bytes = get_file_size(repo_id, filename)
    delete_old_files_until_space(local_base_dir, file_size_bytes)

    logging.info("Downloading {} from {} to {}".format(filename, repo_id, local_base_dir))

    start_time = time.time()
    file_path = await asyncio.to_thread(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        local_dir=local_base_dir,
        token=os.environ.get('HF_TOKEN')
    )
    end_time = time.time()
    
    update_access_time(file_path)
    
    download_time = end_time - start_time
    file_size_mb = file_size_bytes / (1024 * 1024)
    download_rate_mbps = file_size_mb / download_time if download_time > 0 else 0
    
    logging.info("Successfully downloaded model to: {}".format(file_path))
    logging.info("Download size: {:.2f} MB".format(file_size_mb))
    logging.info("Download time: {:.2f} seconds".format(download_time))
    logging.info("Download rate: {:.2f} MB/s".format(download_rate_mbps))
    return file_path


async def download_models(obj, repo_id):
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
            if data.strip():
                found.add(data)

        return found

    all_strings = find_all_strings(obj)

    if not all_strings:
        logging.info("No strings found in object")
        return []

    logging.info("Found {} string(s) to check against repo".format(len(all_strings)))

    repo_files = get_repo_files(repo_id)

    local_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    downloaded_paths = []

    for potential_filename in all_strings:
        matches = [f for f in repo_files if os.path.basename(f) == potential_filename]

        if not matches:
            if potential_filename.endswith('.safetensors'):
                logging.warning("No matches found in repo for safetensors file: {}".format(potential_filename))
            continue

        if len(matches) > 1:
            logging.warning("Multiple matches found for '{}': {}".format(potential_filename, matches))
            logging.warning("Using first match: {}".format(matches[0]))

        repo_file_path = matches[0]

        local_file_path = os.path.join(local_base_dir, repo_file_path)

        if os.path.exists(local_file_path):
            update_access_time(local_file_path)
            logging.info("File already exists locally: {}".format(local_file_path))
            downloaded_paths.append(local_file_path)
        else:
            logging.info("File not found locally, downloading: {}".format(repo_file_path))
            downloaded_path = await download_huggingface_model(repo_id, repo_file_path)
            downloaded_paths.append(downloaded_path)

    return downloaded_paths

