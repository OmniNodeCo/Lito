//go:build windows

package main

import (
	"os/exec"
	"syscall"
)

// createNoWindow (CREATE_NO_WINDOW) stops Windows from opening a console
// window for child processes. The app is built with -H=windowsgui (no
// console of its own), so without this every `python server.py` it spawns
// would flash a terminal window - and a crash-looping engine would flash
// one every few seconds.
const createNoWindow = 0x08000000

// hideWindow marks a child process to run without a console window.
func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: createNoWindow}
}
