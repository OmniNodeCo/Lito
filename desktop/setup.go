package main

// In-app setup: when the desktop app finds no AI engine next to itself,
// it can download the engine-only zip (LitoEngine-<platform>.zip from the
// latest GitHub release) and unpack it - no installer, no Python, no
// terminal. The supervisor notices the new engine within seconds and
// starts it. The URL can be overridden with LITO_ENGINE_URL (testing,
// mirrors).

import (
	"archive/zip"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Setup states reported by /api/desktop/setup.
const (
	setupIdle        = "idle"
	setupDownloading = "downloading"
	setupExtracting  = "extracting"
	setupDone        = "done"
	setupError       = "error"
)

const defaultRepo = "OmniNodeCo/Lito"

type setupManager struct {
	mu      sync.Mutex
	state   string
	message string
	got     int64
	total   int64
	running bool
}

func (s *setupManager) set(state, message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = state
	s.message = message
	log.Printf("setup: %s (%s)", state, message)
}

func (s *setupManager) progress(got, total int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.got = got
	s.total = total
}

func (s *setupManager) snapshot() (state, message string, got, total int64, running bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state, s.message, s.got, s.total, s.running
}

func (s *setupManager) fail(msg string) {
	s.mu.Lock()
	s.running = false
	s.state = setupError
	s.message = msg
	s.mu.Unlock()
	log.Printf("setup: error (%s)", msg)
}

// begin starts a setup run; false if one is already in progress.
func (s *setupManager) begin() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running {
		return false
	}
	s.running = true
	s.state = setupDownloading
	s.message = "Downloading the AI engine…"
	s.got, s.total = 0, 0
	return true
}

// engineZipURL is where the engine-only zip for this platform lives.
func engineZipURL() (string, error) {
	if u := os.Getenv("LITO_ENGINE_URL"); u != "" {
		return u, nil
	}
	var platform string
	switch {
	case runtime.GOOS == "windows":
		platform = "Windows"
	case runtime.GOOS == "linux":
		platform = "Linux"
	case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
		platform = "macOS-apple"
	default:
		return "", fmt.Errorf("no prebuilt engine for %s/%s - download a LitoDesktop bundle or run from source",
			runtime.GOOS, runtime.GOARCH)
	}
	return fmt.Sprintf("https://github.com/%s/releases/latest/download/LitoEngine-%s.zip",
		defaultRepo, platform), nil
}

// run downloads and installs the engine next to the app. Run it in a
// goroutine.
func (s *setupManager) run(exeDir string) {
	url, err := engineZipURL()
	if err != nil {
		s.fail(err.Error())
		return
	}
	log.Printf("setup: downloading %s", url)

	client := &http.Client{Timeout: 15 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		s.fail("download failed: " + err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		s.fail(fmt.Sprintf("download failed: HTTP %d (%s)", resp.StatusCode, url))
		return
	}

	tmp := filepath.Join(exeDir, ".LitoEngine.download")
	out, err := os.Create(tmp)
	if err != nil {
		s.fail("cannot write next to the app: " + err.Error())
		return
	}

	// Stream with progress reporting.
	total := resp.ContentLength // may be -1 (unknown)
	var got int64
	buf := make([]byte, 64*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := out.Write(buf[:n]); werr != nil {
				out.Close()
				os.Remove(tmp)
				s.fail("writing the download failed: " + werr.Error())
				return
			}
			got += int64(n)
			s.progress(got, total)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			out.Close()
			os.Remove(tmp)
			s.fail("download interrupted: " + rerr.Error())
			return
		}
	}
	out.Close()
	if got == 0 {
		os.Remove(tmp)
		s.fail("the download was empty")
		return
	}

	s.set(setupExtracting, "Unpacking the AI engine…")
	if err := extractZip(tmp, exeDir); err != nil {
		os.Remove(tmp)
		os.RemoveAll(filepath.Join(exeDir, ".LitoEngine.new"))
		s.fail("unpacking failed: " + err.Error())
		return
	}
	os.Remove(tmp)

	s.mu.Lock()
	s.running = false
	s.state = setupDone
	s.message = "AI engine installed - starting it…"
	s.mu.Unlock()
	log.Printf("setup: engine installed in %s", exeDir)
}

// extractZip unpacks archive (whose entries live under LitoEngine/)
// into dest via a staging folder, so a half-written engine never appears.
func extractZip(archive, dest string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()

	stage := filepath.Join(dest, ".LitoEngine.new")
	os.RemoveAll(stage)
	if err := os.MkdirAll(stage, 0o755); err != nil {
		return err
	}

	for _, f := range zr.File {
		name := filepath.ToSlash(f.Name)          // normalize separators
		name = strings.TrimPrefix(name, "/")      // tolerate absolute entries
		clean := filepath.Clean(name)
		if strings.HasPrefix(clean, "..") || clean == "." {
			continue // zip-slip guard
		}
		target := filepath.Join(stage, clean)
		if !strings.HasPrefix(target, stage+string(filepath.Separator)) {
			continue
		}

		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}
		w, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(w, rc); err != nil {
			rc.Close()
			w.Close()
			return err
		}
		rc.Close()
		w.Close()

		// Restore permissions (zip stores them; the engine binary must
		// stay executable). Windows ignores the mode.
		perm := f.FileInfo().Mode().Perm()
		if perm == 0 {
			perm = 0o644
		}
		if strings.HasSuffix(clean, ".exe") || !strings.Contains(filepath.Base(clean), ".") {
			perm |= 0o100 // executable bit for the engine binary
		}
		_ = os.Chmod(target, perm)
	}

	// Move the unpacked engine into place.
	final := filepath.Join(dest, "LitoEngine")
	if _, err := os.Stat(final); err == nil {
		os.RemoveAll(final)
	}
	if err := os.Rename(filepath.Join(stage, "LitoEngine"), final); err != nil {
		os.RemoveAll(stage)
		return err
	}
	os.RemoveAll(stage)
	return nil
}
