// Command lito-desktop is the Lito desktop app, Ollama-style: it starts
// the Python backend (server.py), serves the embedded web UI on a local
// port and proxies /api/* to the backend.
//
// Build:
//
//	go build -trimpath -ldflags "-s -w" .
//
// Run (from a Lito source checkout with Python + numpy installed):
//
//	./lito-desktop            # or lito-desktop.exe on Windows
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
	"syscall"
	"time"
)

//go:embed ui
var uiFS embed.FS

const defaultPort = 5486

func main() {
	port := flag.Int("port", defaultPort, "port for the desktop UI")
	backend := flag.String("backend", "", "backend URL (skip launching server.py)")
	pythonBin := flag.String("python", "", "python executable for the backend")
	noBrowser := flag.Bool("no-browser", false, "do not open a browser window")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	backendURL := strings.TrimRight(*backend, "/")

	// ------------------------------------------------------------------
	// Launch the Python backend unless one was given.
	// ------------------------------------------------------------------
	if backendURL == "" {
		py := *pythonBin
		if py == "" {
			py = findPython()
		}
		serverPy := findServerPy()
		if py == "" || serverPy == "" {
			fmt.Println("Lito desktop could not find Python or server.py.")
			fmt.Println()
			fmt.Println("Run it from a Lito source checkout with Python + numpy")
			fmt.Println("installed, or start the backend yourself and point at it:")
			fmt.Println()
			fmt.Println("    python server.py")
			fmt.Println("    lito-desktop -backend http://127.0.0.1:5486")
			fmt.Println()
			os.Exit(1)
		}

		bport, err := freePort()
		if err != nil {
			log.Fatalf("finding a free backend port: %v", err)
		}
		backendURL = fmt.Sprintf("http://127.0.0.1:%d", bport)

		child := exec.Command(py, serverPy,
			"--port", strconv.Itoa(bport), "--no-browser")
		child.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			log.Fatalf("starting the backend: %v", err)
		}
		defer func() {
			if child.Process != nil {
				_ = child.Process.Kill()
			}
			_ = child.Wait()
		}()
		log.Printf("backend: %s %s --port %d", py, serverPy, bport)

		if err := waitReady(ctx, backendURL, 90*time.Second); err != nil {
			log.Fatalf("backend did not come up: %v", err)
		}
		log.Printf("backend ready at %s", backendURL)
	}

	target, err := url.Parse(backendURL)
	if err != nil {
		log.Fatalf("bad backend URL %q: %v", backendURL, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusBadGateway)
		fmt.Fprintf(w, `{"error": "backend unavailable: %v"}`, e)
	}

	// ------------------------------------------------------------------
	// Serve the embedded UI + proxy the API.
	// ------------------------------------------------------------------
	sub, err := fs.Sub(uiFS, "ui")
	if err != nil {
		log.Fatalf("embedded UI: %v", err)
	}
	files := http.FileServer(http.FS(sub))

	mux := http.NewServeMux()
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
		log.Fatalf("listening: %v", err)
	}

	srv := &http.Server{Handler: mux} // bound via ln
	uiURL := "http://" + addr
	log.Printf("Lito desktop UI: %s  (Ctrl+C to quit)", uiURL)
	if !*noBrowser {
		go openBrowser(uiURL)
	}

	go func() {
		<-ctx.Done()
		log.Printf("shutting down…")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	log.Printf("bye!")
}

// waitReady polls the backend's /api/version until it answers.
func waitReady(ctx context.Context, backendURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}
	var lastErr error
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		resp, err := client.Get(backendURL + "/api/version")
		if err == nil {
			resp.Body.Close()
			return nil
		}
		lastErr = err
		time.Sleep(400 * time.Millisecond)
	}
	return fmt.Errorf("timeout after %s, last error: %v", timeout, lastErr)
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

// findPython locates a Python interpreter on PATH.
func findPython() string {
	candidates := []string{"python", "python3"}
	if runtime.GOOS == "windows" {
		candidates = []string{"python", "py", "python3"}
	}
	for _, c := range candidates {
		if _, err := exec.LookPath(c); err == nil {
			return c
		}
	}
	return ""
}

// findServerPy looks for server.py next to the executable, near it, or in
// the working directory (covers running from a source checkout, from
// desktop/, from dist/ inside the repo, or a bundled release zip).
func findServerPy() string {
	exe, err := os.Executable()
	if err == nil {
		exeDir := filepath.Dir(exe)
		candidates := []string{
			filepath.Join(exeDir, "server.py"),
			filepath.Join(exeDir, "..", "server.py"),
			filepath.Join(exeDir, "..", "..", "server.py"),
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
