//go:build !windows

package main

import "os/exec"

// hideWindow is a no-op elsewhere: only Windows opens console windows for
// child processes.
func hideWindow(cmd *exec.Cmd) {}
