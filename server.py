#!/usr/bin/env python3
"""
Lito API server - the local backend for the Lito desktop app.

Exposes the brain, the model manager (version database), the extensions
and training over a localhost JSON API, Ollama-style:

    GET  /api/version          app + model status (poll while loading)
    GET  /api/models           local models (trained + downloaded)
    GET  /api/releases         version database (GitHub Releases)
    POST /api/models/use       {name} or {path} - switch the active model
    POST /api/models/delete    {name}
    POST /api/download         {tag} - start downloading a release
    GET  /api/download/status  {active, done, total, error}
    GET  /api/extensions       list + enabled state
    POST /api/extensions       {name, enabled}
    POST /api/search           {enabled}
    POST /api/chat             {message} -> {response}
    POST /api/reset            reset the conversation
    GET  /api/info             model / search / knowledge info
    POST /api/train            {epochs, d_model, ..., classic} start training
    GET  /api/train/status     ?since=N - log lines after N + state

It also serves the desktop web UI (desktop/ui/) when present, so running
just `python server.py` gives the whole app in a browser. The Go desktop
app (desktop/) embeds the same UI and proxies /api/* to this server.

Run:
    python server.py [--port 5486] [--dir <model_dir>] [--no-search] [--no-browser]

Standard library only - no extra dependencies.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.brain import AIBrain
from src.extensions import ExtensionManager
from src.model_manager import ModelManager, format_size
from src.version import APP_NAME, __version__, format_version

UI_DIR = os.path.join(ROOT, 'desktop', 'ui')

# Training options the API accepts (name -> flag name for train.py).
TRAIN_INT_FIELDS = ('epochs', 'd_model', 'n_layers', 'n_heads', 'd_ff',
                    'vocab_size', 'batch_size', 'seq_len')
TRAIN_FLOAT_FIELDS = ('lr',)


class LitoServer:
    """Shared state + background workers behind the HTTP API."""

    def __init__(self, model_dir=None, search_enabled=True):
        self.mm = ModelManager(models_dir=os.path.join(ROOT, 'models'))
        self.ext = ExtensionManager()
        self.search_enabled = search_enabled
        self.brain = None
        self.brain_error = None
        self.brain_dir = None
        # Guards thinking (long) against model swaps (fast).
        self._brain_lock = threading.Lock()
        self._load_lock = threading.Lock()
        # Download state (single active download).
        self.download_state = {'active': False, 'tag': '', 'done': 0,
                               'total': 0, 'error': None, 'model': None}
        self._download_lock = threading.Lock()
        # Training state (single run).
        self.train_state = {'running': False, 'lines': [],
                            'returncode': None, 'started_at': ''}
        self._train_lock = threading.Lock()
        self.startup_dir = model_dir or os.path.join(ROOT, 'checkpoints')

    # ------------------------------------------------------------------
    # Brain lifecycle
    # ------------------------------------------------------------------

    def start_background_load(self):
        def worker():
            try:
                self.load_brain(self.startup_dir)
            except Exception as e:  # pragma: no cover - defensive
                self.brain_error = str(e)
        threading.Thread(target=worker, daemon=True).start()

    def load_brain(self, model_dir):
        """Build + initialize a brain for model_dir and make it active."""
        with self._load_lock:
            brain = AIBrain(model_dir=model_dir,
                            search_enabled=self.search_enabled,
                            extensions=self.ext)
            brain.initialize()
            with self._brain_lock:
                self.brain = brain
                self.brain_dir = model_dir
                self.brain_error = None
            return brain

    def status(self):
        brain = self.brain
        return {
            'name': APP_NAME,
            'version': format_version(__version__),
            'loading': brain is None and self.brain_error is None,
            'loaded': bool(brain and brain.loaded),
            'error': self.brain_error,
            'model': os.path.basename(self.brain_dir) if self.brain_dir else '',
            'params': getattr(brain.model, '_param_count', 0) if brain and brain.loaded else 0,
            'search': bool(brain.search.enabled) if brain else self.search_enabled,
        }

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, message):
        """Answer a chat message (or a slash command)."""
        message = (message or '').strip()
        if not message:
            return {'response': ''}
        if message.startswith('/'):
            return {'response': self.handle_command(message)}
        with self._brain_lock:
            brain = self.brain
            if brain is None:
                if self.brain_error:
                    return {'response':
                            f'(model failed to load: {self.brain_error})'}
                return {'response': 'Still loading the model - try again in '
                                    'a moment.', 'loading': True}
            reply = brain.think(message)
            return {'response': reply}

    def handle_command(self, message):
        """Slash commands, mirroring the terminal chat."""
        parts = message.split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ''
        brain = self.brain

        if cmd in ('/quit', '/exit'):
            return '(the desktop app keeps running - just close the window)'
        if cmd == '/help':
            lines = ['Commands: /help /info /reset /extensions /generate']
            for c, help_text, name in self.ext.command_help():
                lines.append(f'  {c:12s} - {help_text}  [{name} extension]')
            return '\n'.join(lines)
        if cmd == '/info':
            s = self.status()
            return (f"{s['name']} {s['version']}\n"
                    f"Model: {s['model'] or '-'} "
                    f"({'loaded' if s['loaded'] else 'not loaded'})\n"
                    f"Parameters: {s['params']:,}\n"
                    f"Web search: {'on' if s['search'] else 'off'}")
        if cmd == '/reset':
            if brain:
                brain.reset_conversation()
            return 'Conversation reset.'
        if cmd == '/extensions':
            return self.extensions_text(args)
        if cmd == '/generate':
            if brain and brain.loaded and args:
                out = brain._generate_with_model(args, max_tokens=100)
                return out or '(no output)'
            return 'Neural generation unavailable (model not loaded).'
        reply = self.ext.handle_command(cmd, args, brain)
        if reply:
            return reply
        return f'Unknown command: {cmd}'

    def extensions_text(self, args=''):
        args = (args or '').strip().lower()
        if args.startswith(('on ', 'off ')):
            action, name = args.split(' ', 1)
            if self.ext.set_enabled(name, action == 'on'):
                return f'Extension "{name}" {action}.'
            return f'No extension named "{name}".'
        lines = []
        for ext in self.ext.extensions:
            mark = 'on ' if self.ext.is_enabled(ext.name) else 'off'
            lines.append(f'[{mark}] {ext.name} v{ext.version} - '
                         f'{ext.description}')
        return '\n'.join(lines) or 'No extensions installed.'

    # ------------------------------------------------------------------
    # Download (version database)
    # ------------------------------------------------------------------

    def start_download(self, tag):
        """Start downloading release `tag` in the background."""
        with self._download_lock:
            if self.download_state['active']:
                return {'error': 'a download is already running'}
            self.download_state.update(active=True, tag=str(tag), done=0,
                                       total=0, error=None, model=None)
        try:
            releases = self.mm.available_releases()
            release = next((r for r in releases if r.tag == tag), None)
        except Exception as e:
            with self._download_lock:
                self.download_state.update(active=False, error=str(e))
            return {'error': f'could not list releases: {e}'}
        if release is None:
            with self._download_lock:
                self.download_state.update(active=False,
                                           error=f'unknown version {tag}')
            return {'error': f'unknown version {tag}'}

        def progress(done, total):
            with self._download_lock:
                self.download_state['done'] = done
                self.download_state['total'] = total

        def worker():
            try:
                info = self.mm.download(release, progress)
                with self._download_lock:
                    self.download_state.update(active=False, model=info.name)
            except Exception as e:
                with self._download_lock:
                    self.download_state.update(active=False, error=str(e))

        threading.Thread(target=worker, daemon=True).start()
        return {'started': True, 'tag': str(tag)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def start_training(self, params):
        with self._train_lock:
            if self.train_state['running']:
                return {'error': 'training is already running'}
            if getattr(sys, 'frozen', False):
                # Frozen engine: no python/train.py on disk - re-run this
                # executable in trainer mode (see the --lito-train shim
                # in main()).
                cmd = [sys.executable, '--lito-train']
            else:
                cmd = [sys.executable, os.path.join(ROOT, 'train.py')]
            for key in TRAIN_INT_FIELDS:
                try:
                    value = int(params.get(key))
                except (TypeError, ValueError):
                    continue
                cmd += [f'--{key}', str(value)]
            for key in TRAIN_FLOAT_FIELDS:
                try:
                    value = float(params.get(key))
                except (TypeError, ValueError):
                    continue
                cmd += [f'--{key}', repr(value)]
            if params.get('classic'):
                cmd.append('--classic')
            try:
                proc = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    env=dict(os.environ, PYTHONUNBUFFERED='1'))
            except Exception as e:
                return {'error': f'failed to start training: {e}'}
            self.train_state.update(running=True, lines=[], returncode=None,
                                    started_at=time.strftime('%Y-%m-%d %H:%M'))

        def reader():
            for line in proc.stdout:
                with self._train_lock:
                    self.train_state['lines'].append(line.rstrip('\n'))
            code = proc.wait()
            with self._train_lock:
                self.train_state['running'] = False
                self.train_state['returncode'] = code
            if code == 0:
                # Annotate the trained model so it shows up nicely.
                self.mm.write_info(os.path.join(ROOT, 'checkpoints'),
                                   source='trained',
                                   trained_at=time.strftime('%Y-%m-%d'))

        threading.Thread(target=reader, daemon=True).start()
        return {'started': True, 'command': ' '.join(cmd)}

    def train_status(self, since=0):
        with self._train_lock:
            lines = self.train_state['lines']
            return {
                'running': self.train_state['running'],
                'returncode': self.train_state['returncode'],
                'total': len(lines),
                'lines': lines[max(0, int(since)):],
            }


def make_handler(server_state: LitoServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = f'{APP_NAME}/{__version__}'
        state = server_state

        # --------------------------------------------------------------
        def log_message(self, fmt, *args):
            pass  # keep the console quiet; errors still raise

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            try:
                n = int(self.headers.get('Content-Length') or 0)
                if n:
                    return json.loads(self.rfile.read(n).decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                pass
            return {}

        # --------------------------------------------------------------
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/api/version':
                self._json(self.state.status())
            elif u.path == '/api/models':
                self._json({'models': [
                    {'name': m.name, 'path': m.path, 'source': m.source,
                     'version': m.version, 'vocab_size': m.vocab_size,
                     'trained_at': m.trained_at}
                    for m in self.state.mm.local_models()]})
            elif u.path == '/api/releases':
                try:
                    releases = self.state.mm.available_releases()
                    self._json({'releases': [
                        {'tag': r.tag, 'name': r.name, 'date': r.published_at,
                         'size': r.asset_size, 'size_text': format_size(r.asset_size),
                         'notes': r.notes_first_line}
                        for r in releases]})
                except Exception as e:
                    self._json({'releases': [], 'error': str(e)})
            elif u.path == '/api/download/status':
                with self.state._download_lock:
                    self._json(dict(self.state.download_state))
            elif u.path == '/api/extensions':
                self._json({'extensions': [
                    {'name': e.name, 'version': e.version,
                     'description': e.description,
                     'commands': e.commands,
                     'enabled': self.state.ext.is_enabled(e.name)}
                    for e in self.state.ext.extensions]})
            elif u.path == '/api/info':
                brain = self.state.brain
                self._json({
                    'name': APP_NAME,
                    'version': format_version(__version__),
                    'model': os.path.basename(self.state.brain_dir or ''),
                    'loaded': bool(brain and brain.loaded),
                    'params': getattr(brain.model, '_param_count', 0) if brain and brain.loaded else 0,
                    'vocab_size': self.state.mm._describe_npz(self.state.brain_dir).get('vocab_size', '') if self.state.brain_dir else '',
                    'search': bool(brain.search.enabled) if brain else self.state.search_enabled,
                    'knowledge_version': format_version(brain.dictionary.knowledge_version) if brain else '',
                    'learned_words': list(brain.dictionary.learned_words) if brain else [],
                })
            elif u.path == '/api/train/status':
                q = parse_qs(u.query)
                since = int(q.get('since', ['0'])[0])
                self._json(self.state.train_status(since))
            elif u.path.startswith('/api/'):
                self._json({'error': f'unknown endpoint {u.path}'}, 404)
            else:
                self._static(u.path)

        # --------------------------------------------------------------
        def do_POST(self):
            u = urlparse(self.path)
            data = self._body()
            st = self.state

            if u.path == '/api/chat':
                message = str(data.get('message', ''))
                if not message.strip():
                    self._json({'error': 'empty message'}, 400)
                else:
                    self._json(st.chat(message))
            elif u.path == '/api/reset':
                if st.brain:
                    st.brain.reset_conversation()
                self._json({'ok': True})
            elif u.path == '/api/search':
                enabled = bool(data.get('enabled'))
                st.search_enabled = enabled
                if st.brain:
                    st.brain.search.enabled = enabled
                self._json({'ok': True, 'search': enabled})
            elif u.path == '/api/extensions':
                name = str(data.get('name', ''))
                enabled = bool(data.get('enabled'))
                if st.ext.set_enabled(name, enabled):
                    self._json({'ok': True, 'name': name, 'enabled': enabled})
                else:
                    self._json({'error': f'no extension named {name!r}'}, 404)
            elif u.path == '/api/models/use':
                target = str(data.get('name') or data.get('path') or '')
                models = st.mm.local_models()
                model = next((m for m in models
                              if m.name == target or m.path == target), None)
                if model is None:
                    self._json({'error': f'no model named {target!r}'}, 404)
                    return
                try:
                    st.load_brain(model.path)
                except Exception as e:
                    self._json({'error': f'failed to load: {e}'}, 500)
                    return
                self._json({'ok': True, **st.status()})
            elif u.path == '/api/models/delete':
                name = str(data.get('name', ''))
                if st.mm.delete(name):
                    if st.brain_dir and os.path.basename(st.brain_dir) == name:
                        with st._brain_lock:
                            st.brain = None
                            st.brain_error = None
                    self._json({'ok': True})
                else:
                    self._json({'error': f'could not delete {name!r}'}, 400)
            elif u.path == '/api/download':
                self._json(st.start_download(str(data.get('tag', ''))))
            elif u.path == '/api/train':
                self._json(st.start_training(data))
            elif u.path.startswith('/api/'):
                self._json({'error': f'unknown endpoint {u.path}'}, 404)
            else:
                self._json({'error': 'not found'}, 404)

        # --------------------------------------------------------------
        # Static UI (so `python server.py` works standalone)
        # --------------------------------------------------------------
        def _static(self, path):
            import mimetypes
            name = path.lstrip('/') or 'index.html'
            name = os.path.normpath(name).lstrip('/')
            if '..' in name.split(os.sep):
                self._json({'error': 'bad path'}, 400)
                return
            full = os.path.join(UI_DIR, name)
            if os.path.isdir(full):
                full = os.path.join(full, 'index.html')
            if not os.path.isfile(full):
                # SPA-style fallback: unknown paths get the app shell.
                full = os.path.join(UI_DIR, 'index.html')
                if not os.path.isfile(full):
                    self._json({'error': 'desktop UI not found '
                                         '(desktop/ui/ is missing)'}, 404)
                    return
            ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
            try:
                with open(full, 'rb') as f:
                    body = f.read()
            except OSError:
                self._json({'error': 'read error'}, 500)
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    # Frozen-engine helper: a PyInstaller-packaged engine has no separate
    # python or train.py, so the API server re-runs ITS OWN executable to
    # train (see start_training): LitoEngine --lito-train <train args>.
    if len(sys.argv) > 1 and sys.argv[1] == '--lito-train':
        sys.argv = ['train.py'] + sys.argv[2:]
        from train import main as train_main
        return train_main()

    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} API server (local backend for the desktop app)')
    parser.add_argument('--port', type=int, default=5486,
                        help='port to listen on (default 5486)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='interface to bind (default 127.0.0.1; use '
                             '0.0.0.0 only on machines you trust)')
    parser.add_argument('--dir', default=None,
                        help='model directory to load (default: checkpoints)')
    parser.add_argument('--no-search', action='store_true',
                        help='disable web search')
    parser.add_argument('--no-browser', action='store_true',
                        help='do not open a browser window')
    args = parser.parse_args()

    state = LitoServer(model_dir=args.dir,
                       search_enabled=not args.no_search)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True
    url = f'http://127.0.0.1:{httpd.server_address[1]}'
    print(f'{APP_NAME} {format_version(__version__)} API server')
    print(f'  listening on {args.host}:{httpd.server_address[1]}')
    print(f'  model dir: {os.path.basename(state.startup_dir)}')
    print(f'  press Ctrl+C to stop')
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    state.start_background_load()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
        httpd.shutdown()


if __name__ == '__main__':
    main()
