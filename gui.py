#!/usr/bin/env python3
"""
Lito GUI - an Ollama-style desktop app for Lito.

    - Chat with your local model (knowledge base, dictionary, web search,
      neural generation and optional extensions)
    - Model manager: browse the version database (GitHub Releases),
      download models, switch between them, delete them
    - Train a new model from scratch (runs in the background)
    - Toggle extensions and web search on the fly

Run it with:

    python gui.py [--dir <model_dir>] [--no-search]

Requires only Python's bundled tkinter (no extra packages).
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

# Make the repo root importable no matter where we launch from.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# When frozen (PyInstaller onefile), bundled read-only data (src, data,
# extensions, checkpoints) is extracted into a temporary dir and ROOT
# points there. Files the user creates at runtime (downloaded models)
# belong next to the real executable instead, so they survive restarts.
FROZEN = bool(getattr(sys, 'frozen', False))
APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if FROZEN else ROOT

from src.brain import AIBrain
from src.extensions import ExtensionManager
from src.model_manager import ModelManager, format_size
from src.version import APP_NAME, __version__, format_version

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
SIDEBAR_BG = '#1e2430'
SIDEBAR_FG = '#e8eaf0'
SIDEBAR_MUTED = '#9aa3b5'
ACCENT = '#4f8cff'
CHAT_BG = '#ffffff'
USER_COLOR = '#1a56c4'
LITO_COLOR = '#0f7b3d'
MUTED = '#6b7280'


class LitoApp(tk.Tk):
    """The main Lito window: model sidebar + chat pane."""

    def __init__(self, model_dir: str = None, search_enabled: bool = True):
        super().__init__()
        self.title(f"{APP_NAME} {format_version(__version__)}")
        self.geometry('1080x700')
        self.minsize(880, 580)

        self.mm = ModelManager(models_dir=os.path.join(APP_DIR, 'models'))
        self.ext = ExtensionManager()
        self.q: "queue.Queue" = queue.Queue()
        self.brain: AIBrain = None
        self.brain_dir = None
        self.train_proc = None
        # Default model: the bundled/repo checkpoints dir (absolute, so it
        # works no matter the current working directory).
        self.startup_dir = model_dir or os.path.join(ROOT, 'checkpoints')

        self._build_style()
        self._build_ui(search_enabled)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(80, self._poll)
        self.after(50, lambda: self._load_brain(self.startup_dir))

    # ------------------------------------------------------------------
    # UI construction (all on the main thread)
    # ------------------------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        base = tkfont.nametofont('TkDefaultFont')
        self.font_bold = base.copy(); self.font_bold.configure(weight='bold')
        self.font_big = base.copy(); self.font_big.configure(size=15, weight='bold')
        self.font_caption = base.copy()
        self.font_caption.configure(size=9, weight='bold')
        self.font_chat = tkfont.nametofont('TkTextFont').copy()
        self.font_chat.configure(size=11)

    def _build_ui(self, search_enabled):
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        main.add(self._build_sidebar(main, search_enabled), weight=0)
        main.add(self._build_chat_pane(main), weight=1)

    def _build_sidebar(self, parent, search_enabled):
        side = tk.Frame(parent, bg=SIDEBAR_BG, width=280)
        side.pack_propagate(False)

        header = tk.Frame(side, bg=SIDEBAR_BG)
        header.pack(fill=tk.X, padx=14, pady=(14, 6))
        tk.Label(header, text='◆ LITO', bg=SIDEBAR_BG, fg=ACCENT,
                 font=self.font_big).pack(side=tk.LEFT)
        tk.Label(header, text=format_version(__version__), bg=SIDEBAR_BG,
                 fg=SIDEBAR_MUTED).pack(side=tk.LEFT, padx=(8, 0), pady=(6, 0))

        # --- Models -----------------------------------------------------
        tk.Label(side, text='MODELS', bg=SIDEBAR_BG, fg=SIDEBAR_MUTED,
                 font=self.font_caption).pack(anchor='w', padx=14, pady=(10, 2))
        list_wrap = tk.Frame(side, bg=SIDEBAR_BG)
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=14)
        scroll = ttk.Scrollbar(list_wrap)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.model_list = tk.Listbox(
            list_wrap, bg='#171c26', fg=SIDEBAR_FG, selectbackground=ACCENT,
            selectforeground='white', relief='flat', highlightthickness=0,
            font=self.font_chat, activestyle='none')
        self.model_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.model_list.yview)
        self.model_list.config(yscrollcommand=scroll.set)
        self.model_list.bind('<Double-Button-1>',
                             lambda e: self._use_selected_model())

        btns = tk.Frame(side, bg=SIDEBAR_BG)
        btns.pack(fill=tk.X, padx=14, pady=8)
        self.download_btn = tk.Button(
            btns, text='⬇  Download…', command=self._open_browser,
            bg=ACCENT, fg='white', relief='flat', padx=10, pady=4,
            activebackground='#3d74e0', activeforeground='white')
        self.download_btn.pack(side=tk.LEFT)
        tk.Button(btns, text='⟳', command=self._refresh_models, bg='#171c26',
                  fg=SIDEBAR_FG, relief='flat', padx=8, pady=4,
                  activebackground='#232a38').pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(btns, text='🗑', command=self._delete_selected_model,
                  bg='#171c26', fg=SIDEBAR_FG, relief='flat', padx=8, pady=4,
                  activebackground='#232a38').pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(btns, text='⚙ Train…', command=self._open_train_dialog,
                  bg='#171c26', fg=SIDEBAR_FG, relief='flat', padx=10, pady=4,
                  activebackground='#232a38').pack(side=tk.LEFT, padx=(6, 0))

        # --- Extensions ---------------------------------------------------
        tk.Label(side, text='EXTENSIONS', bg=SIDEBAR_BG, fg=SIDEBAR_MUTED,
                 font=self.font_caption).pack(anchor='w', padx=14, pady=(8, 2))
        ext_wrap = tk.Frame(side, bg=SIDEBAR_BG)
        ext_wrap.pack(fill=tk.X, padx=14)
        self.ext_vars = {}
        for ext in self.ext.extensions:
            var = tk.BooleanVar(value=self.ext.is_enabled(ext.name))
            cb = tk.Checkbutton(
                ext_wrap, text=f'{ext.name}', variable=var, bg=SIDEBAR_BG,
                fg=SIDEBAR_FG, selectcolor='#171c26', activebackground=SIDEBAR_BG,
                activeforeground=SIDEBAR_FG, relief='flat',
                font=self.font_chat, anchor='w',
                command=lambda name=ext.name, v=var: self._toggle_ext(name, v))
            cb.pack(fill=tk.X, anchor='w')
            self.ext_vars[ext.name] = var

        # --- Settings -----------------------------------------------------
        tk.Label(side, text='SETTINGS', bg=SIDEBAR_BG, fg=SIDEBAR_MUTED,
                 font=self.font_caption).pack(anchor='w', padx=14, pady=(12, 2))
        self.search_var = tk.BooleanVar(value=search_enabled)
        tk.Checkbutton(side, text='Web search', variable=self.search_var,
                       bg=SIDEBAR_BG, fg=SIDEBAR_FG, selectcolor='#171c26',
                       activebackground=SIDEBAR_BG, activeforeground=SIDEBAR_FG,
                       font=self.font_chat, anchor='w',
                       command=self._toggle_search).pack(anchor='w', padx=14)

        return side

    def _build_chat_pane(self, parent):
        pane = tk.Frame(parent, bg=CHAT_BG)

        # Header
        header = tk.Frame(pane, bg='#f3f4f6')
        header.pack(fill=tk.X)
        self.status_lbl = tk.Label(header, text='Loading…', bg='#f3f4f6',
                                   fg=MUTED, anchor='w', padx=14, pady=8,
                                   font=self.font_bold)
        self.status_lbl.pack(fill=tk.X)

        # Chat log
        from tkinter.scrolledtext import ScrolledText
        self.chat = ScrolledText(pane, wrap=tk.WORD, bg=CHAT_BG, relief='flat',
                                 padx=16, pady=12, font=self.font_chat,
                                 state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_configure('user', foreground=USER_COLOR,
                                font=self.font_bold)
        self.chat.tag_configure('lito', foreground=LITO_COLOR)
        self.chat.tag_configure('muted', foreground=MUTED,
                                font=self.font_caption)

        # Input
        input_wrap = tk.Frame(pane, bg='#f3f4f6')
        input_wrap.pack(fill=tk.X)
        self.input = tk.Text(input_wrap, height=3, relief='flat', bg='white',
                             font=self.font_chat, padx=12, pady=8)
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6),
                        pady=10)
        self.input.bind('<Return>', self._on_enter)
        self.input.bind('<Shift-Return>', lambda e: None)
        self.send_btn = tk.Button(input_wrap, text='Send ➤', command=self._send,
                                  bg=ACCENT, fg='white', relief='flat',
                                  padx=18, pady=6, font=self.font_bold,
                                  activebackground='#3d74e0',
                                  activeforeground='white')
        self.send_btn.pack(side=tk.RIGHT, padx=(0, 12), pady=10)
        return pane

    # ------------------------------------------------------------------
    # Startup / background workers
    # ------------------------------------------------------------------

    def _load_brain(self, model_dir: str):
        """Load (or reload) the brain in a background thread."""
        if model_dir in (None, ''):
            model_dir = 'checkpoints'
        self._set_status(f'Loading {os.path.basename(model_dir)}…')
        self.send_btn.config(state=tk.DISABLED)

        def worker():
            try:
                brain = AIBrain(model_dir=model_dir,
                                search_enabled=self.search_var.get(),
                                extensions=self.ext)
                brain.initialize()
                self.q.put(('brain_ready', brain, model_dir))
            except Exception as e:
                self.q.put(('brain_error', str(e), model_dir))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == 'brain_ready':
                    self._on_brain_ready(msg[1], msg[2])
                elif kind == 'brain_error':
                    self._set_status(f'Failed to load model: {msg[1]}')
                    self._append_chat('lito', f'⚠ Failed to load model: {msg[1]}\n'
                                              'You can still chat, but the neural '
                                              'model is unavailable.\n')
                    self.send_btn.config(state=tk.NORMAL)
                elif kind == 'reply':
                    self._append_chat('lito', msg[1] + '\n\n')
                    self.send_btn.config(state=tk.NORMAL)
                    self.input.focus_set()
                elif kind == 'releases':
                    self._on_releases(msg[1], msg[2])
                elif kind == 'download_progress':
                    self._on_download_progress(msg[1], msg[2])
                elif kind == 'download_done':
                    self._on_download_done(msg[1], msg[2])
                elif kind == 'train_log':
                    self._on_train_log(msg[1])
                elif kind == 'train_done':
                    self._on_train_done(msg[1])
        except queue.Empty:
            pass
        self.after(80, self._poll)

    # ------------------------------------------------------------------
    # Sidebar actions
    # ------------------------------------------------------------------

    def _refresh_models(self):
        self.model_list.delete(0, tk.END)
        self._models = self.mm.local_models()
        active = os.path.basename(self.brain_dir) if self.brain_dir else None
        for m in self._models:
            tag = '  ✦' if m.name == active else ''
            label = f'{m.name}{tag}  ({m.source})'
            self.model_list.insert(tk.END, label)

    def _selected_model(self):
        sel = self.model_list.curselection()
        if not sel:
            return None
        return self._models[sel[0]] if hasattr(self, '_models') and sel[0] < len(self._models) else None

    def _use_selected_model(self):
        model = self._selected_model()
        if model:
            self._load_brain(model.path)

    def _delete_selected_model(self):
        model = self._selected_model()
        if not model:
            return
        if model.name == 'checkpoints':
            messagebox.showinfo(APP_NAME, 'The checkpoints folder is your own '
                                          'trained model - delete it manually '
                                          'if you really want to.')
            return
        if not messagebox.askyesno(APP_NAME, f'Delete model "{model.name}"?\n'
                                             f'This cannot be undone.'):
            return
        if self.mm.delete(model.name):
            if self.brain_dir and os.path.basename(self.brain_dir) == model.name:
                self.brain = None
                self.brain_dir = None
            self._refresh_models()
            self._set_status('Model deleted.')
        else:
            messagebox.showerror(APP_NAME, f'Could not delete "{model.name}".')

    def _toggle_ext(self, name, var):
        self.ext.set_enabled(name, var.get())
        state = 'enabled' if var.get() else 'disabled'
        self._append_chat('muted', f'Extension "{name}" {state}.\n')

    def _toggle_search(self):
        if self.brain is not None:
            self.brain.search.enabled = self.search_var.get()
        state = 'on' if self.search_var.get() else 'off'
        self._append_chat('muted', f'Web search {state}.\n')

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _set_status(self, text):
        self.status_lbl.config(text=text)

    def _on_brain_ready(self, brain, model_dir):
        self.brain = brain
        self.brain_dir = model_dir
        params = getattr(brain.model, '_param_count', 0) if brain.loaded else 0
        loaded = 'neural model loaded' if brain.loaded else 'knowledge base only'
        ptxt = f' • {params:,} params' if params else ''
        self._set_status(f'Model: {os.path.basename(model_dir)} — {loaded}{ptxt}')
        self.send_btn.config(state=tk.NORMAL)
        self._refresh_models()
        self._append_chat('muted', f'{APP_NAME} {format_version(__version__)} — '
                                   'type /help for commands.\n\n')

    def _append_chat(self, tag, text):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, text, tag)
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _on_enter(self, event):
        # Enter sends (Shift+Enter inserts a newline, handled by default).
        self._send()
        return 'break'

    def _send(self):
        text = self.input.get('1.0', tk.END).strip()
        if not text:
            return
        self.input.delete('1.0', tk.END)
        self._append_chat('user', f'You: {text}\n\n')

        if text.startswith('/'):
            self._handle_command(text)
            return
        if self.brain is None:
            self._append_chat('lito', 'Still loading the model… one moment.\n\n')
            return

        self.send_btn.config(state=tk.DISABLED)

        def worker():
            try:
                reply = self.brain.think(text)
            except Exception as e:
                reply = f'(error: {e})'
            self.q.put(('reply', reply))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_command(self, text):
        parts = text.split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ''
        if cmd in ('/quit', '/exit'):
            self._on_close()
        elif cmd == '/help':
            lines = ['Commands: /help /info /reset /extensions /quit']
            for c, h, name in self.ext.command_help():
                lines.append(f'  {c:12s} - {h}  [{name} extension]')
            self._append_chat('lito', '\n'.join(lines) + '\n\n')
        elif cmd == '/info':
            b = self.brain
            info = (f'{APP_NAME} {format_version(__version__)}\n'
                    f'Model dir: {self.brain_dir}\n'
                    f'Neural model: {"loaded" if b and b.loaded else "not loaded"}\n'
                    f'Web search: {"on" if b and b.search.enabled else "off"}')
            self._append_chat('lito', info + '\n\n')
        elif cmd == '/reset':
            if self.brain:
                self.brain.reset_conversation()
            self._append_chat('lito', 'Conversation reset. ✅\n\n')
        elif cmd == '/extensions':
            self._append_chat('lito', self._extensions_text(args) + '\n\n')
        else:
            reply = self.ext.handle_command(cmd, args, self.brain)
            if reply:
                self._append_chat('lito', reply + '\n\n')
            else:
                self._append_chat('lito', f'Unknown command: {cmd}\n\n')

    def _extensions_text(self, args=''):
        args = args.strip().lower()
        if args.startswith('on ') or args.startswith('off '):
            action, name = args.split(' ', 1)
            if self.ext.set_enabled(name, action == 'on'):
                if name in self.ext_vars:
                    self.ext_vars[name].set(action == 'on')
                return f'Extension "{name}" {action}.'
            return f'No extension named "{name}".'
        lines = []
        for ext in self.ext.extensions:
            mark = '✅' if self.ext.is_enabled(ext.name) else '⬜'
            lines.append(f'{mark} {ext.name} v{ext.version} — {ext.description}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Version database browser (download dialog)
    # ------------------------------------------------------------------

    def _open_browser(self):
        ModelBrowser(self)

    def _on_releases(self, dialog, releases):
        dialog.show_releases(releases)

    def _on_download_progress(self, done, total):
        if hasattr(self, '_dl_bar') and self._dl_bar.winfo_exists():
            frac = (done / total) if total else 0
            self._dl_bar['value'] = frac * 100
            self._dl_lbl.config(text=f'{format_size(done)}'
                                     f'{" / " + format_size(total) if total else ""}')

    def _on_download_done(self, info, error):
        if hasattr(self, '_dl_bar') and self._dl_bar.winfo_exists():
            self._dl_bar['value'] = 100
        if error:
            self._append_chat('lito', f'⚠ Download failed: {error}\n\n')
            messagebox.showerror(APP_NAME, f'Download failed:\n{error}')
            return
        self._append_chat('muted', f'Downloaded {info.name} '
                                   f'({info.version or "model"}).\n')
        self._refresh_models()
        # Auto-select and activate the new model.
        for i, m in enumerate(getattr(self, '_models', [])):
            if m.name == info.name:
                self.model_list.selection_clear(0, tk.END)
                self.model_list.selection_set(i)
                self._load_brain(m.path)
                break

    # ------------------------------------------------------------------
    # Training dialog hooks
    # ------------------------------------------------------------------

    def _open_train_dialog(self):
        TrainDialog(self)

    def _on_train_log(self, line):
        if hasattr(self, '_train_log') and self._train_log.winfo_exists():
            self._train_log.config(state=tk.NORMAL)
            self._train_log.insert(tk.END, line + '\n')
            self._train_log.see(tk.END)
            self._train_log.config(state=tk.DISABLED)

    def _on_train_done(self, code):
        if code == 0:
            self._append_chat('muted', 'Training finished. ✅\n')
            self._refresh_models()
            self._load_brain(os.path.join(ROOT, 'checkpoints'))
        else:
            self._append_chat('muted', f'Training exited with code {code}.\n')

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _on_close(self):
        try:
            if self.train_proc and self.train_proc.poll() is None:
                if not messagebox.askyesno(
                        APP_NAME, 'Training is still running. Quit anyway?'):
                    return
                self.train_proc.terminate()
        except Exception:
            pass
        self.destroy()


class ModelBrowser(tk.Toplevel):
    """Browse the version database (GitHub Releases) and download models."""

    def __init__(self, app: LitoApp):
        super().__init__(app)
        self.app = app
        self.title('Download models — version database')
        self.geometry('660x420')
        self.transient(app)

        ttk.Label(self, text='Models published on GitHub Releases',
                  font=app.font_bold).pack(anchor='w', padx=12, pady=(12, 4))
        ttk.Label(self, text=f'Repository: {app.mm.repo}',
                  foreground=MUTED).pack(anchor='w', padx=12)

        columns = ('version', 'name', 'date', 'size')
        self.tree = ttk.Treeview(self, columns=columns, show='headings')
        for col, heading, width in (('version', 'Version', 90),
                                    ('name', 'Release', 220),
                                    ('date', 'Date', 100),
                                    ('size', 'Size', 90)):
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.tree.bind('<Double-1>', lambda e: self._download())

        bottom = tk.Frame(self)
        bottom.pack(fill=tk.X, padx=12, pady=(0, 12))
        self.status = ttk.Label(bottom, text='Loading version database…',
                                foreground=MUTED)
        self.status.pack(side=tk.LEFT)
        self.dl_btn = ttk.Button(bottom, text='⬇ Download selected',
                                 command=self._download, state=tk.DISABLED)
        self.dl_btn.pack(side=tk.RIGHT)

        app._dl_bar = ttk.Progressbar(bottom, length=180, mode='determinate')
        app._dl_lbl = self.status
        self.releases = []
        self.after(50, self._fetch)

    def _fetch(self):
        def worker():
            try:
                releases = self.app.mm.available_releases()
                self.app.q.put(('releases', self, releases))
            except Exception as e:
                self.app.q.put(('releases', self,
                                RuntimeError(f'Could not reach GitHub: {e}')))

        threading.Thread(target=worker, daemon=True).start()

    def show_releases(self, releases):
        if isinstance(releases, Exception):
            self.status.config(text=str(releases))
            return
        self.releases = releases
        for r in releases:
            self.tree.insert('', tk.END, values=(
                r.tag, r.name, r.published_at, format_size(r.asset_size)))
        if releases:
            self.status.config(text=f'{len(releases)} versions found — '
                                    'select one and click Download.')
            self.dl_btn.config(state=tk.NORMAL)
            self.tree.selection_set(self.tree.get_children()[0])
        else:
            self.status.config(text='No published models yet. Train your own '
                                    'with the Train button!')

    def _download(self):
        sel = self.tree.selection()
        if not sel or not self.releases:
            return
        release = self.releases[self.tree.index(sel[0])]
        self.dl_btn.config(state=tk.DISABLED)
        self.app._dl_bar.pack(side=tk.RIGHT, padx=(0, 10))
        self.status.config(text=f'Downloading {release.tag}…')

        def progress(done, total):
            self.app.q.put(('download_progress', done, total))

        def worker():
            try:
                info = self.app.mm.download(release, progress)
                self.app.q.put(('download_done', info, None))
            except Exception as e:
                self.app.q.put(('download_done', None, str(e)))

        threading.Thread(target=worker, daemon=True).start()


class TrainDialog(tk.Toplevel):
    """Configure and run a training run (python train.py) in the background."""

    FIELDS = [
        ('epochs', '30'), ('d_model', '128'), ('n_layers', '4'),
        ('n_heads', '4'), ('d_ff', '512'), ('vocab_size', '2048'),
        ('batch_size', '4'), ('seq_len', '64'), ('lr', '0.0003'),
    ]

    def __init__(self, app: LitoApp):
        super().__init__(app)
        self.app = app
        self.title('Train a new model')
        self.geometry('620x520')
        self.transient(app)

        form = ttk.Frame(self)
        form.pack(fill=tk.X, padx=12, pady=12)
        self.entries = {}
        for i, (key, default) in enumerate(self.FIELDS):
            ttk.Label(form, text=key).grid(row=i // 3, column=(i % 3) * 2,
                                           sticky='w', padx=6, pady=4)
            e = ttk.Entry(form, width=9)
            e.insert(0, default)
            e.grid(row=i // 3, column=(i % 3) * 2 + 1, padx=6, pady=4)
            self.entries[key] = e
        self.classic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text='Classic architecture (v1.0: learned positions, '
                                   'LayerNorm, untied)',
                        variable=self.classic_var).grid(
            row=3, column=0, columnspan=6, sticky='w', padx=6, pady=6)

        ttk.Label(self, foreground=MUTED,
                  text='Runs "python train.py" in the background - output '
                       'appears below. The finished model lands in '
                       'checkpoints/ and is loaded automatically.').pack(
            fill=tk.X, padx=12)

        from tkinter.scrolledtext import ScrolledText
        self.log = ScrolledText(self, height=14, state=tk.DISABLED,
                                font=('Consolas', 9))
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.app._train_log = self.log

        ttk.Button(self, text='▶ Start training',
                   command=self._start).pack(pady=(0, 12))

    def _start(self):
        if FROZEN:
            self._log_line('Training needs the Python source code and is not '
                           'available in the packaged app.')
            self._log_line('Use Download to fetch trained models, or run '
                           '"python train.py" from a source checkout.')
            return

        cmd = [sys.executable, os.path.join(ROOT, 'train.py')]
        for key, entry in self.entries.items():
            value = entry.get().strip()
            if value:
                cmd += [f'--{key}', value]
        if self.classic_var.get():
            cmd.append('--classic')

        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, f'$ {" ".join(cmd)}\n\n')
        self.log.config(state=tk.DISABLED)

        try:
            self.app.train_proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            self._log_line(f'Failed to start: {e}')
            return

        def reader():
            proc = self.app.train_proc
            for line in proc.stdout:
                self.app.q.put(('train_log', line.rstrip()))
            proc.wait()
            self.app.q.put(('train_done', proc.returncode))

        threading.Thread(target=reader, daemon=True).start()

    def _log_line(self, line):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, line + '\n')
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)


def main():
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} GUI - chat, download and train models')
    parser.add_argument('--dir', default=None,
                        help='model directory to load (default: checkpoints)')
    parser.add_argument('--no-search', action='store_true',
                        help='disable web search at startup')
    args = parser.parse_args()

    app = LitoApp(model_dir=args.dir,
                  search_enabled=not args.no_search)
    app.mainloop()


if __name__ == '__main__':
    main()
