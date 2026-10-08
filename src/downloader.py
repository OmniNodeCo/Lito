"""
Model downloader - downloads trained models and tokenizers from GitHub
Releases, GitHub Actions workflow artifacts, or custom URLs.
"""

import os
import sys
import json
import zipfile
import shutil
import urllib.request
import urllib.error
import ssl


def _format_size(bytes_num: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"


def download_file_with_progress(url: str, dest_path: str, headers: dict = None) -> bool:
    """Download a file with an interactive progress bar."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers or {'User-Agent': 'Lito-Downloader'})

    try:
        with urllib.request.urlopen(req, context=ctx) as response, open(dest_path, 'wb') as out_file:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192

            print(f"Downloading from: {url}")
            if total_size > 0:
                print(f"File size: {_format_size(total_size)}")

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    bar_len = 30
                    filled_len = int(bar_len * downloaded // total_size)
                    bar = '=' * filled_len + '-' * (bar_len - filled_len)
                    sys.stdout.write(f"\r[{bar}] {percent:5.1f}% ({_format_size(downloaded)}/{_format_size(total_size)})")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloaded: {_format_size(downloaded)}")
                    sys.stdout.flush()

            print("\nDownload complete!")
            return True
    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\nDownload error: {e}")
        return False


def get_git_remote_repo() -> str:
    """Attempt to detect repository (owner/repo) from local git config."""
    try:
        import subprocess
        out = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.DEVNULL)
        url = out.decode().strip()
        # Handle SSH and HTTPS urls
        if 'github.com' in url:
            if url.startswith('git@github.com:'):
                repo = url.replace('git@github.com:', '').replace('.git', '')
            else:
                parts = url.split('github.com/')[-1].replace('.git', '').split('/')
                repo = f"{parts[0]}/{parts[1]}"
            return repo
    except Exception:
        pass
    return os.environ.get('GITHUB_REPOSITORY', '')


def download_model_from_release(repo: str = '', tag: str = 'latest',
                                save_dir: str = 'checkpoints', token: str = None) -> bool:
    """
    Download trained model (model.zip) from latest GitHub Release.
    """
    os.makedirs(save_dir, exist_ok=True)
    repo = repo or get_git_remote_repo()

    if not repo:
        repo = input("Enter GitHub repository (owner/repo, e.g. username/Lito): ").strip()

    if not repo or '/' not in repo:
        print("Invalid repository format. Must be 'owner/repo'.")
        return False

    headers = {'User-Agent': 'Lito-Downloader'}
    if token or os.environ.get('GITHUB_TOKEN'):
        tok = token or os.environ.get('GITHUB_TOKEN')
        headers['Authorization'] = f'Bearer {tok}'

    api_url = f"https://api.github.com/repos/{repo}/releases/latest" if tag == 'latest' else f"https://api.github.com/repos/{repo}/releases/tags/{tag}"

    print(f"Fetching latest release metadata from: {api_url}")
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch release info (HTTP {e.code}): {e.reason}")
        if e.code == 404:
            print(f"No release found at {repo}. Make sure a release with 'model.zip' exists.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    # Find model.zip asset or direct individual files
    assets = data.get('assets', [])
    download_url = None
    asset_name = None

    for asset in assets:
        if asset['name'] in ['model.zip', 'checkpoints.zip', 'trained-model.zip']:
            download_url = asset['browser_download_url']
            asset_name = asset['name']
            break

    if not download_url:
        print("No 'model.zip' found in release assets.")
        # Try downloading individual files
        files_to_get = ['best_model.npz', 'tokenizer.json']
        found_any = False
        for f in files_to_get:
            for asset in assets:
                if asset['name'] == f:
                    dest = os.path.join(save_dir, f)
                    if download_file_with_progress(asset['browser_download_url'], dest, headers):
                        found_any = True
        return found_any

    # Download zip file
    zip_path = os.path.join(save_dir, asset_name)
    success = download_file_with_progress(download_url, zip_path, headers)
    if not success:
        return False

    # Extract contents
    print(f"Extracting {asset_name} to {save_dir}/...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        os.remove(zip_path)
        print("Model extraction complete!")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def download_model_from_url(url: str, save_dir: str = 'checkpoints') -> bool:
    """Download model zip from any direct HTTP/HTTPS URL."""
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, 'model_temp.zip')
    if download_file_with_progress(url, zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
            os.remove(zip_path)
            print("Model extracted successfully!")
            return True
        except zipfile.BadZipFile:
            # Maybe it was directly best_model.npz
            shutil.move(zip_path, os.path.join(save_dir, 'best_model.npz'))
            print("Saved as best_model.npz")
            return True
    return False