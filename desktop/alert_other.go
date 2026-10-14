//go:build !windows

package main

// alertUser is a no-op on platforms where the app runs in a terminal and
// log output is already visible.
func alertUser(title, message string) {}
