"""
Model manager + version database for Lito.

- Local models live in `models/<name>/` (each holding a `lito-chat.gguf`
  and an info.json written after download), plus a `lito-chat.gguf`
  exported next to the code by `python export_gguf.py`.
- The *version database* is the repository's GitHub Releases: the
  "Build Lito" workflow trains and publishes a `lito-chat.gguf` for
  every version.

Used by the API server / desktop app but has no UI dependencies itself.
"""

import json
import os
import re
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional

DEFAULT_REPO = 'OmniNodeCo/Lito'
GITHUB_API = 'https://api.github.com'
MODEL_FILENAME = 'lito-chat.gguf'
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

    def _describe_gguf(self, model_dir: str) -> dict:
        """Read the metadata inside a lito-chat.gguf (cheap, no weights)."""
        path = os.path.join(model_dir, MODEL_FILENAME)
        if not os.path.exists(path):
            return {}
        try:
            from .gguf import read_gguf_metadata
            meta, shapes = read_gguf_metadata(path)
            params = 1
            total = 0
            for ne, _t in shapes.values():
                n = 1
                for d in ne:
                    n *= d
                total += n
            return {
                'params': total,
                'vocab_size': meta.get('llama.vocab_size',
                                       len(meta.get('tokenizer.ggml.tokens',
                                                    []))),
                'model_name': meta.get('general.name', ''),
                'context_length': meta.get('llama.context_length', ''),
                'arch': 'llama',
            }
        except Exception:
            return {}

    def _model_info(self, name: str, model_dir: str, source: str) -> ModelInfo:
        info = self._read_info(model_dir)
        config = self._describe_gguf(model_dir) if not info else {}
        params = info.get('params', config.get('params', ''))
        if isinstance(params, int):
            params = f'{params:,}'
        return ModelInfo(
            name=name,
            path=model_dir,
            source=info.get('source', source),
            version=info.get('version', ''),
            params=str(params),
            vocab_size=str(info.get('vocab_size', config.get('vocab_size', ''))),
            trained_at=info.get('trained_at', info.get('downloaded_at', '')),
        )

    def local_models(self) -> List[ModelInfo]:
        """All usable models on disk (downloaded + locally exported)."""
        models: List[ModelInfo] = []
        if os.path.isdir(self.models_dir):
            for name in sorted(os.listdir(self.models_dir)):
                d = os.path.join(self.models_dir, name)
                if os.path.isdir(d) and os.path.isfile(
                        os.path.join(d, MODEL_FILENAME)):
                    models.append(self._model_info(name, d, 'local'))
        # A model exported next to the code (train_big.py + export_gguf.py)
        # counts as a locally trained model.
        exported = os.path.join(_ROOT, MODEL_FILENAME)
        if os.path.isfile(exported):
            models.append(self._model_info('lito-chat (exported)',
                                           _ROOT, 'trained'))
        return models

    def model_path(self, info: ModelInfo) -> str:
        """The GGUF file behind a ModelInfo."""
        if os.path.isfile(os.path.join(info.path, MODEL_FILENAME)):
            return os.path.join(info.path, MODEL_FILENAME)
        return info.path  # already a file path

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
        """List the version database (GitHub releases with a GGUF model)."""
        url = f'{GITHUB_API}/repos/{self.repo}/releases?per_page=30'
        data = self._fetch_json(url)
        releases = []
        for rel in data if isinstance(data, list) else []:
            if rel.get('draft'):
                continue
            asset = None
            for a in rel.get('assets', []):
                if a.get('name') == MODEL_FILENAME:
                    asset = a
                    break
            if asset is None:
                ggufs = [a for a in rel.get('assets', []) or []
                         if str(a.get('name', '')).endswith('.gguf')]
                asset = ggufs[0] if ggufs else None
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
        """Download a release's lito-chat.gguf into models/<tag>/."""
        tag = release.tag or 'model'
        target = os.path.join(self.models_dir, tag)
        os.makedirs(target, exist_ok=True)
        dest = os.path.join(target, MODEL_FILENAME)
        self._download_file(release.asset_url, dest + '.part', progress_cb)
        os.replace(dest + '.part', dest)

        info = {
            'source': 'local',
            'version': tag,
            'downloaded_at': release.published_at,
        }
        config = self._describe_gguf(target)
        if config:
            info['vocab_size'] = config.get('vocab_size', '')
            info['arch'] = config.get('arch', 'llama')
        self.write_info(target, **info)
        return self._model_info(tag, target, 'local')
