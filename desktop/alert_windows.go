//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

// alertUser shows a native Windows message box. The Windows build of the
// app has no console window, so this is the only visible channel for
// fatal startup errors (double-clicking must never fail in silence).
func alertUser(title, message string) {
	t, err := syscall.UTF16PtrFromString(title)
	if err != nil {
		return
	}
	m, err := syscall.UTF16PtrFromString(message)
	if err != nil {
		return
	}
	// MB_OK | MB_ICONERROR | MB_TOPMOST = 0x10 | 0x100000
	user32 := syscall.NewLazyDLL("user32.dll")
	messageBoxW := user32.NewProc("MessageBoxW")
	_, _, _ = messageBoxW.Call(0, uintptr(unsafe.Pointer(m)), uintptr(unsafe.Pointer(t)), 0x10|0x40000)
}
