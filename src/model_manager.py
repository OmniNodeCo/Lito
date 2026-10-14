"""
Model manager + version database for Lito.

- Local models live in `models/<name>/` (each holding best_model.npz,
  tokenizer.json and an info.json written after download/training).
- The *version database* is the repository's GitHub Releases: the
  "Release Lito" workflow publishes a model.zip for every version.
- `checkpoints/` (created by `python train.py`) is listed as a local
  model too, so freshly trained models show up next to downloaded ones.

Used by the API server / desktop app but has no UI dependencies itself.
"""

import io
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, List, Optional

DEFAULT_REPO = 'OmniNodeCo/Lito'
GITHUB_API = 'https://api.github.com'
_USER_AGENT = 'Lito-ModelManager'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelInfo:
    """A model available on this machine."""
    name: str
    path: str
    source: str = 'local'        # 'local' (downloaded) | 'trained'
    version: str = ''
    params: str = ''
    vocab_size: str = ''
    trained_at: str = ''


@dataclass
class ReleaseInfo:
    """One entry of the version database (a GitHub release)."""
    tag: str
    name: str
    published_at: str
    asset_url: str = ''
    asset_size: int = 0
    notes_first_line: str = ''


def format_size(num_bytes: int) -> str:
    if not num_bytes:
        return ''
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num_bytes < 1024 or unit == 'GB':
            return f'{num_bytes:.1f} {unit}' if unit != 'B' else f'{num_bytes} B'
        num_bytes /= 1024
    return f'{num_bytes:.1f} GB'


class ModelManager:
    """Discover local models, list + download releases, delete models."""

    def __init__(self, models_dir: str = None, repo: str = DEFAULT_REPO,
                 token: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(_ROOT, 'models')
        self.repo = repo
        self.token = token

    # ------------------------------------------------------------------
    # Local models
    # ------------------------------------------------------------------

    def _read_info(self, model_dir: str) -> dict:
        try:
            with open(os.path.join(model_dir, 'info.json'), 'r',
                      encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _describe_npz(self, model_dir: str) -> dict:
        """Read the config stored inside best_model.npz (cheap metadata)."""
        for fname in ('best_model.npz', 'final_model.npz'):
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                import numpy as np
                loaded = np.load(path, allow_pickle=True)
                if 'config' in loaded:
                    return json.loads(str(loaded['config']))
            except Exception:
                return {}
        return {}

    def _model_info(self, name: str, model_dir: str, source: str) -> ModelInfo:
        info = self._read_info(model_dir)
        config = self._describe_npz(model_dir) if not info else {}
        return ModelInfo(
            name=name,
            path=model_dir,
            source=info.get('source', source),
            version=info.get('version', ''),
            params=info.get('params', ''),
            vocab_size=str(info.get('vocab_size', config.get('vocab_size', ''))),
            trained_at=info.get('trained_at', info.get('downloaded_at', '')),
        )

    def local_models(self) -> List[ModelInfo]:
        """All usable models on disk (downloaded + trained)."""
        models: List[ModelInfo] = []
        if os.path.isdir(self.models_dir):
            for name in sorted(os.listdir(self.models_dir)):
                d = os.path.join(self.models_dir, name)
                if os.path.isdir(d) and (
                        os.path.exists(os.path.join(d, 'best_model.npz'))
                        or os.path.exists(os.path.join(d, 'final_model.npz'))):
                    models.append(self._model_info(name, d, 'local'))
        # The training output directory (train.py) is a model too.
        ckpt = os.path.join(_ROOT, 'checkpoints')
        if os.path.isdir(ckpt) and (
                os.path.exists(os.path.join(ckpt, 'best_model.npz'))
                or os.path.exists(os.path.join(ckpt, 'final_model.npz'))):
            models.append(self._model_info('checkpoints', ckpt, 'trained'))
        return models

    def write_info(self, model_dir: str, **fields) -> None:
        info = self._read_info(model_dir)
        info.update(fields)
        try:
            with open(os.path.join(model_dir, 'info.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(info, f, indent=2)
        except OSError:
            pass

    def delete(self, name: str) -> bool:
        """Delete a downloaded model directory. Refuses unsafe names."""
        if not re.fullmatch(r'[\w.-]+', name or '') or name in ('.', '..'):
            return False
        for model in self.local_models():
            if model.name == name and model.source != 'trained':
                shutil.rmtree(model.path, ignore_errors=True)
                return True
        return False

    # ------------------------------------------------------------------
    # Version database (GitHub Releases)
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        headers = {
            'User-Agent': _USER_AGENT,
            'Accept': 'application/vnd.github+json',
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    def _fetch_json(self, url: str, headers: Optional[dict] = None) -> dict:
        """GET a URL and parse JSON. Overridable in tests."""
        req = urllib.request.Request(url, headers=headers or self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def _download_file(self, url: str, dest: str,
                       progress_cb: Optional[Callable[[int, int], None]] = None,
                       headers: Optional[dict] = None) -> str:
        """Stream a file to dest. Overridable in tests."""
        req = urllib.request.Request(url, headers=headers or self._headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get('Content-Length') or 0)
            done = 0
            with open(dest, 'wb') as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total)
        return dest

    def available_releases(self) -> List[ReleaseInfo]:
        """List the version database (GitHub releases with a model asset)."""
        url = f'{GITHUB_API}/repos/{self.repo}/releases?per_page=30'
        data = self._fetch_json(url)
        releases = []
        for rel in data if isinstance(data, list) else []:
            if rel.get('draft'):
                continue
            asset = None
            for a in rel.get('assets', []):
                if a.get('name') == 'model.zip':
                    asset = a
                    break
            if asset is None:
                zips = [a for a in rel.get('assets', []) or []
                        if str(a.get('name', '')).endswith('.zip')]
                asset = zips[0] if zips else None
            if asset is None:
                continue
            body = (rel.get('body') or '').strip().splitlines()
            releases.append(ReleaseInfo(
                tag=rel.get('tag_name', ''),
                name=rel.get('name') or rel.get('tag_name', ''),
                published_at=(rel.get('published_at') or '')[:10],
                asset_url=asset.get('browser_download_url', ''),
                asset_size=int(asset.get('size') or 0),
                notes_first_line=body[0][:120] if body else '',
            ))
        releases.sort(key=lambda r: r.tag, reverse=True)
        return releases

    def download(self, release: ReleaseInfo,
                 progress_cb: Optional[Callable[[int, int], None]] = None) -> ModelInfo:
        """Download a release's model.zip into models/<tag>/."""
        tag = release.tag or 'model'
        target = os.path.join(self.models_dir, tag)
        os.makedirs(self.models_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, 'model.zip')
            self._download_file(release.asset_url, zip_path, progress_cb)
            os.makedirs(target, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    if member.endswith('/'):
                        continue
                    out_name = os.path.basename(member)
                    if not out_name:
                        continue
                    with zf.open(member) as src, \
                            open(os.path.join(target, out_name), 'wb') as dst:
                        shutil.copyfileobj(src, dst)

        info = {
            'source': 'local',
            'version': tag,
            'downloaded_at': release.published_at,
        }
        config = self._describe_npz(target)
        if config:
            info['vocab_size'] = config.get('vocab_size', '')
            info['arch'] = (f"{config.get('pos_encoding', 'learned')}/"
                            f"{config.get('norm', 'layer')}")
        self.write_info(target, **info)
        return self._model_info(tag, target, 'local')
