// Command lito-desktop is the Lito desktop app, Ollama-style: it serves
// the embedded web UI on a local port and runs the AI engine as a child
// process, proxying /api/* to it. The engine is the frozen LitoEngine
// binary that ships in the release bundles (Python + numpy included, so
// nothing needs to be installed); when running from a source checkout it
// falls back to launching `python server.py`.
//
// The UI always starts, even when Python or server.py can't be found:
// the page then shows a banner explaining what is missing, and the app
// keeps looking for the engine in the background so it recovers on its
// own. Diagnostics go to lito-desktop.log next to the executable (the
// Windows build has no console window).
//
// Build:
//
//	go build -trimpath -ldflags "-s -w" .
//
// Run (from a Lito release bundle or source checkout):
//
//	./lito-desktop            # or LitoDesktop-Windows.exe
//
// Flags:
//
//	-port N        UI port (default 5486)
//	-backend URL   talk to an already-running backend instead of
//	               launching python server.py
//	-python PATH   python executable to use (default: python/python3/py)
//	-no-browser    don't open the browser
package main

import (
	"context"
	"embed"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed ui
var uiFS embed.FS

const defaultPort = 5486

// Backend states reported by /api/desktop/status.
const (
	stateStarting = "starting" // engine launching, not answering yet
	stateReady    = "ready"    // engine answered /api/version
	stateMissing  = "missing"  // python or server.py not found (still retrying)
	stateError    = "error"    // engine crashed / never came up (retrying)
)

// engineDownHost is a host that never answers: the proxy sends requests
// there when no engine URL is known yet, so they fail fast with the
// standard 502 JSON instead of hanging.
const engineDownHost = "engine-not-started.invalid"

// backendManager owns the Python engine process and its state.
type backendManager struct {
	mu    sync.Mutex
	state string
	msg   string

	backendURL string // what the proxy talks to ("" until the engine spawns)
	pythonBin  string // from -python flag ("" = auto-detect)
	external   bool   // true when -backend was given: probe, don't spawn

	child     *exec.Cmd
	exited    bool          // true once the child process is gone
	childDone chan struct{} // closed when the child exits
}

func (m *backendManager) set(state, msg string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.state != state || m.msg != msg {
		log.Printf("engine state: %s (%s)", state, msg)
	}
	m.state = state
	m.msg = msg
}

func (m *backendManager) snapshot() (string, string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state, m.msg
}

// currentBackend returns the URL the proxy should target right now.
func (m *backendManager) currentBackend() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.backendURL
}

// run manages the engine for the whole life of the app: discover python +
// server.py, spawn the engine, wait for it to answer, restart it when it
// dies, and keep retrying discovery while it is missing. It only returns
// when ctx is done.
func (m *backendManager) run(ctx context.Context) {
	if m.external {
		m.runExternal(ctx)
		return
	}
	for ctx.Err() == nil {
		if !m.hasLiveChild() {
			m.spawn(ctx)
		}
		m.monitor(ctx)
		// Whatever happened - engine missing, dead on arrival, or died
		// after running fine - pause before the next attempt. This is
		// the single throttle point: a broken engine can never respawn
		// faster than once every 5 seconds (spawn()'s failure paths do
		// not sleep on their own).
		sleepCtx(ctx, 5*time.Second)
	}
	m.stopChild()
}

func (m *backendManager) hasLiveChild() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.child != nil && !m.exited
}

// runExternal just probes a user-provided backend URL.
func (m *backendManager) runExternal(ctx context.Context) {
	m.set(stateStarting, "Connecting to the AI engine at "+m.backendURL+"…")
	for ctx.Err() == nil {
		if probeOK(m.backendURL) {
			m.set(stateReady, "")
		} else {
			m.set(stateError, "The AI engine at "+m.backendURL+" is not answering. Is `python server.py` running?")
		}
		sleepCtx(ctx, 2*time.Second)
	}
}

// spawn looks for the AI engine and starts it: the frozen LitoEngine
// binary that ships in the LitoDesktop bundles first (no Python needed),
// falling back to `python server.py` from a source checkout. When
// nothing usable is found it records a helpful message and returns
// without a child; run() throttles the retries.
func (m *backendManager) spawn(ctx context.Context) {
	m.stopChild()

	engineBin := findEngineBinary()
	serverPy := findServerPy()
	var candidates []string
	if m.pythonBin != "" {
		candidates = []string{m.pythonBin}
	} else {
		candidates = pythonCandidates()
	}

	if engineBin == "" && serverPy == "" {
		if len(candidates) > 0 {
			m.set(stateMissing, "The AI engine was not found. Keep the app inside its unzipped Lito folder from the LitoDesktop zip (LitoDesktop + LitoEngine together), or run it from a Lito source checkout - Lito keeps looking…")
		} else {
			m.set(stateMissing, "The AI engine was not found. Keep the app inside its unzipped Lito folder from the LitoDesktop zip (LitoDesktop + LitoEngine together). Running from source? Install Python 3 with numpy from python.org - Lito keeps looking…")
		}
		return
	}

	// Source mode: the engine runs as `python server.py`, so a working
	// Python with numpy is required. (The frozen engine needs nothing.)
	py := ""
	if engineBin == "" {
		var lastFail string
		for _, c := range candidates {
			if ok, detail := pythonReady(c); ok {
				py = c
				break
			} else {
				lastFail = detail
			}
		}
		if py == "" {
			msg := "Python is installed but the numpy package is missing - open a terminal and run `pip install numpy`, then come back - Lito keeps checking…"
			if len(candidates) == 0 {
				msg = "The AI engine (server.py) was found but Python was not - install Python 3 with numpy from python.org - Lito keeps looking…"
			} else if m.pythonBin != "" {
				msg = "The Python set with -python is not working (" + firstLine(lastFail) + ") - fix the path or leave the flag off. Lito keeps checking…"
			}
			m.set(stateMissing, msg)
			return
		}
	}

	bport, err := freePort()
	if err != nil {
		m.set(stateError, "could not find a free port for the engine: "+err.Error())
		return
	}
	m.mu.Lock()
	m.backendURL = fmt.Sprintf("http://127.0.0.1:%d", bport)
	m.mu.Unlock()

	var child *exec.Cmd
	if engineBin != "" {
		child = exec.Command(engineBin,
			"--host", "127.0.0.1",
			"--port", strconv.Itoa(bport),
			"--no-browser")
		child.Dir = filepath.Dir(engineBin)
	} else {
		child = exec.Command(py, serverPy,
			"--host", "127.0.0.1",
			"--port", strconv.Itoa(bport),
			"--no-browser")
		child.Dir = filepath.Dir(serverPy) // so checkpoints/ & data/ resolve next to server.py
	}
	child.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	child.Stdout = log.Writer()
	child.Stderr = log.Writer()
	hideWindow(child) // never show a console window for the engine
	if err := child.Start(); err != nil {
		m.set(stateError, "could not start the AI engine: "+err.Error())
		return
	}

	done := make(chan struct{})
	m.mu.Lock()
	m.child = child
	m.exited = false
	m.childDone = done
	m.mu.Unlock()

	go func() {
		err := child.Wait()
		log.Printf("engine exited: %v", err)
		m.mu.Lock()
		m.exited = true
		m.mu.Unlock()
		close(done)
	}()

	if engineBin != "" {
		m.set(stateStarting, "Starting the AI engine (LitoEngine)…")
		log.Printf("engine: %s --port %d", engineBin, bport)
	} else {
		m.set(stateStarting, "Starting the AI engine (Python)…")
		log.Printf("engine: %s %s --port %d", py, serverPy, bport)
	}
}

// monitor watches a live engine: flips to ready when it answers, and to
// error when it dies or misses the 90s startup deadline. It returns as
// soon as the engine is gone (run() respawns it) or ctx is done.
func (m *backendManager) monitor(ctx context.Context) {
	deadline := time.Now().Add(90 * time.Second)
	for ctx.Err() == nil {
		m.mu.Lock()
		exited := m.exited
		done := m.childDone
		m.mu.Unlock()
		if done == nil {
			return
		}

		if probeOK(m.currentBackend()) {
			m.set(stateReady, "")
			// Healthy: sit here until the engine dies or the app quits.
			select {
			case <-ctx.Done():
				return
			case <-done:
				m.set(stateError, "The AI engine stopped unexpectedly - restarting it… (details in lito-desktop.log)")
				return
			}
		}
		if exited {
			m.set(stateError, "The AI engine stopped unexpectedly - restarting it… (details in lito-desktop.log)")
			return
		}
		if time.Now().After(deadline) {
			m.set(stateError, "The AI engine did not start within 90s - restarting it… (details in lito-desktop.log)")
			m.stopChild()
			return
		}
		sleepCtx(ctx, 500*time.Millisecond)
	}
}

// stopChild kills the engine process, if any.
func (m *backendManager) stopChild() {
	m.mu.Lock()
	child := m.child
	m.child = nil
	m.mu.Unlock()
	if child != nil && child.Process != nil {
		_ = child.Process.Kill()
	}
}

// probeOK does a quick /api/version request against the backend.
func probeOK(backendURL string) bool {
	if backendURL == "" {
		return false
	}
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(backendURL + "/api/version")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return true
}

func sleepCtx(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
	case <-time.After(d):
	}
}

func main() {
	port := flag.Int("port", defaultPort, "port for the desktop UI")
	backend := flag.String("backend", "", "backend URL (skip launching server.py)")
	pythonBin := flag.String("python", "", "python executable for the backend")
	noBrowser := flag.Bool("no-browser", false, "do not open a browser window")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	setupLogging()
	defer closeLogFile()

	backendURL := strings.TrimRight(*backend, "/")
	if backendURL != "" {
		if u, err := url.Parse(backendURL); err != nil || u.Scheme == "" || u.Host == "" {
			fatalErr(fmt.Sprintf("bad backend URL %q", *backend))
		}
	}

	mgr := &backendManager{
		state:      stateStarting,
		backendURL: backendURL,
		pythonBin:  *pythonBin,
		external:   backendURL != "",
	}

	// Proxy /api/* to wherever the engine currently is. The target is
	// resolved per request: the engine URL is only known once it spawns,
	// and it can change when it is restarted.
	proxy := &httputil.ReverseProxy{
		Director: func(r *http.Request) {
			be := mgr.currentBackend()
			if be == "" {
				r.URL.Scheme = "http"
				r.URL.Host = engineDownHost
				return
			}
			u, err := url.Parse(be)
			if err != nil || u.Host == "" {
				r.URL.Scheme = "http"
				r.URL.Host = engineDownHost
				return
			}
			r.URL.Scheme = u.Scheme
			r.URL.Host = u.Host
			r.Host = u.Host
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, e error) {
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.WriteHeader(http.StatusBadGateway)
			fmt.Fprintf(w, `{"error": "backend unavailable: %v"}`, e)
		},
	}

	// ------------------------------------------------------------------
	// Serve the embedded UI + proxy the API. This happens BEFORE the
	// engine is up: the app always launches, the UI just shows a banner
	// until the engine is ready.
	// ------------------------------------------------------------------
	sub, err := fs.Sub(uiFS, "ui")
	if err != nil {
		fatalErr("embedded UI: " + err.Error())
	}
	files := http.FileServer(http.FS(sub))

	mux := http.NewServeMux()
	mux.HandleFunc("/api/desktop/status", func(w http.ResponseWriter, r *http.Request) {
		state, msg := mgr.snapshot()
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		fmt.Fprintf(w, `{"state": %q, "message": %q}`, state, msg)
	})
	mux.Handle("/api/", proxy)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/")
		if name == "" {
			name = "index.html"
		}
		if _, err := fs.Stat(sub, name); err != nil {
			// Unknown path: serve the app shell (single-page app).
			r.URL.Path = "/"
		}
		files.ServeHTTP(w, r)
	})

	ln, addr, err := listen(*port)
	if err != nil {
		fatalErr("could not listen on any port: " + err.Error())
	}

	srv := &http.Server{Handler: mux} // bound via ln
	uiURL := "http://" + addr
	log.Printf("Lito desktop UI: %s  (Ctrl+C to quit)", uiURL)
	if !*noBrowser {
		go openBrowser(uiURL)
	}

	go mgr.run(ctx)

	go func() {
		<-ctx.Done()
		log.Printf("shutting down…")
		mgr.stopChild()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		fatalErr(err.Error())
	}
	log.Printf("bye!")
}

// setupLogging tees logs to stdout and lito-desktop.log next to the
// executable (the Windows build has no console, so the file is the only
// trace). Best effort: file problems are not fatal.
var logFile io.WriteCloser

func setupLogging() {
	writers := []io.Writer{os.Stdout}
	if dir := exeDir(); dir != "" {
		if f, err := os.OpenFile(filepath.Join(dir, "lito-desktop.log"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644); err == nil {
			writers = append(writers, f)
			logFile = f
		}
	}
	log.SetOutput(io.MultiWriter(writers...))
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
}

func closeLogFile() {
	if logFile != nil {
		_ = logFile.Close()
	}
}

func exeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	return filepath.Dir(exe)
}

// fatalErr reports an unrecoverable problem: log it, and on Windows (no
// console) show a message box so double-clicking the app never dies in
// silence.
func fatalErr(msg string) {
	log.Printf("FATAL: %s", msg)
	alertUser("Lito desktop", msg)
	os.Exit(1)
}

// listen binds 127.0.0.1:port, trying up to 10 consecutive ports if busy.
func listen(port int) (net.Listener, string, error) {
	for p := port; p < port+10; p++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", p))
		if err == nil {
			return ln, ln.Addr().String(), nil
		}
	}
	return nil, "", fmt.Errorf("no free port in %d-%d", port, port+9)
}

// freePort asks the OS for an unused port (tiny race, fine for localhost).
func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	_, portStr, err := net.SplitHostPort(ln.Addr().String())
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(portStr)
}

// pythonCandidates lists Python interpreters visible on PATH.
func pythonCandidates() []string {
	names := []string{"python", "python3"}
	if runtime.GOOS == "windows" {
		names = []string{"python", "py", "python3"}
	}
	var found []string
	seen := map[string]bool{}
	for _, c := range names {
		if path, err := exec.LookPath(c); err == nil && !seen[path] {
			seen[path] = true
			found = append(found, c)
		}
	}
	return found
}

// pythonReady reports whether `py` is a working Python with numpy
// importable - the two things the AI engine needs. Runs hidden (no
// console flash on Windows) and captures output for the error message.
func pythonReady(py string) (bool, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, py, "-c", "import numpy")
	hideWindow(cmd)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return true, ""
	}
	detail := strings.TrimSpace(string(out))
	if ctx.Err() != nil {
		detail = "timed out"
	}
	if detail == "" {
		detail = err.Error()
	}
	return false, detail
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// findEngineBinary locates the frozen AI engine (LitoEngine) that ships
// in the LitoDesktop release bundles: a onedir build (LitoEngine/
// LitoEngine[.exe] + _internal/) or a onefile binary, next to this
// executable, near it, or in the working directory (covers dist/ inside
// a source checkout too).
func findEngineBinary() string {
	name := "LitoEngine"
	if runtime.GOOS == "windows" {
		name = "LitoEngine.exe"
	}
	var dirs []string
	if d := exeDir(); d != "" {
		dirs = append(dirs,
			d,
			filepath.Join(d, "LitoEngine"),
			filepath.Join(d, ".."),
			filepath.Join(d, "..", "LitoEngine"),
			filepath.Join(d, "..", ".."),
			filepath.Join(d, "..", "..", "LitoEngine"),
		)
	}
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, cwd, filepath.Join(cwd, "LitoEngine"))
	}
	for _, dir := range dirs {
		p := filepath.Join(dir, name)
		if fileExists(p) {
			return filepath.Clean(p)
		}
	}
	return ""
}

// findServerPy looks for server.py next to the executable, near it, or in
// the working directory (covers running from a source checkout, from
// desktop/, from dist/ inside the repo, or a bundled release zip).
func findServerPy() string {
	if dir := exeDir(); dir != "" {
		candidates := []string{
			filepath.Join(dir, "server.py"),
			filepath.Join(dir, "..", "server.py"),
			filepath.Join(dir, "..", "..", "server.py"),
		}
		for _, c := range candidates {
			if fileExists(c) {
				return filepath.Clean(c)
			}
		}
	}
	cwd, err := os.Getwd()
	if err == nil {
		candidates := []string{
			filepath.Join(cwd, "server.py"),
			filepath.Join(cwd, "..", "server.py"),
		}
		for _, c := range candidates {
			if fileExists(c) {
				return filepath.Clean(c)
			}
		}
	}
	return ""
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// openBrowser opens the default browser, best-effort.
func openBrowser(u string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", u)
	case "darwin":
		cmd = exec.Command("open", u)
	default:
		cmd = exec.Command("xdg-open", u)
	}
	if err := cmd.Start(); err != nil {
		log.Printf("could not open a browser (%v) - open %s manually", err, u)
	}
}
