package aec

import (
	"fmt"
	"log"
	"os"
	"time"
)

// Saving the learned echo path across restarts (see ExportState).
//
// Loaded when the hardware-reference filter is built, which is when the
// first playback after boot confirms ch8. Saved from the attenuation
// telemetry: after a one-second playback window cancelling at least
// saveMinDb, at most once a day, because this runs for years on eMMC that
// cannot be replaced. Only the hardware path saves: the software tap's
// alignment depends on runtime delay, so its filter would not transfer.
//
// A saved path that does not load (wrong shape, corrupt, a firmware with a
// different filter) costs a cold start, which is what every boot was before.

const (
	// Converged cancellation measured ~20dB and cold 0-2dB on the bench
	// (2026-09-22), so 12dB says the filter has learnt this echo path.
	saveMinDb = 12.0
	// Minimum age before the file is rewritten, by wall clock when the file
	// exists and by uptime within a run.
	saveEvery = 24 * time.Hour
)

// SetStatePath enables saving and loading the echo path at path. Empty
// disables both.
func (c *Canceller) SetStatePath(path string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.statePath = path
}

// nearStatePath is the file for one mic's echo path, beside the main one.
func (c *Canceller) nearStatePath(ch int) string {
	return fmt.Sprintf("%s.ch%d", c.statePath, ch)
}

// loadStateLocked loads the saved echo path into a freshly built
// hardware-path filter, and the per-mic paths into their parking slots
// (SelectNear), so a mic the lock picks cancels from its first frames after a
// restart too. A few small file reads, once per boot.
func (c *Canceller) loadStateLocked() {
	if c.statePath == "" || !c.hwRef || c.st == nil {
		return
	}
	b, err := os.ReadFile(c.statePath)
	switch {
	case err != nil && !os.IsNotExist(err):
		log.Printf("[aec] saved echo path unreadable, starting cold: %v", err)
	case err != nil:
	default:
		// importLocked, not ImportState: the lock is already held.
		if err := c.importLocked(b); err != nil {
			log.Printf("[aec] saved echo path refused, starting cold: %v", err)
		} else {
			log.Printf("[aec] loaded the saved echo path (%d bytes)", len(b))
		}
	}
	loaded := 0
	for ch := 0; ch <= maxNearCh; ch++ {
		b, err := os.ReadFile(c.nearStatePath(ch))
		if err != nil {
			continue
		}
		if err := c.checkLocked(b); err != nil {
			log.Printf("[aec] saved echo path for ch%d refused: %v", ch, err)
			continue
		}
		if c.nearStates == nil {
			c.nearStates = make(map[int][]byte)
		}
		c.nearStates[ch] = b
		loaded++
	}
	if loaded > 0 {
		log.Printf("[aec] loaded the saved echo paths of %d mic(s)", loaded)
	}
}

// maybeSaveLocked is called once per attenuation window. The export is a
// 16KB copy under the lock; the file write runs on its own goroutine so the
// mic goroutine never waits on flash.
func (c *Canceller) maybeSaveLocked(attDb float64, playing bool) {
	if c.statePath == "" || !c.hwRef || !playing || attDb < saveMinDb {
		return
	}
	if c.lastSaveTry.IsZero() || time.Since(c.lastSaveTry) >= saveEvery {
		c.lastSaveTry = time.Now()
		if b, err := c.exportLocked(); err == nil {
			go writeState(c.statePath, b, attDb)
		}
	}
	// The mic this was learnt on gets its own file, on the same terms: the
	// first time it converges, then daily. Seven files at most.
	if c.nearCh < 0 || c.nearCh > maxNearCh {
		return
	}
	if last, ok := c.lastNearSave[c.nearCh]; ok && time.Since(last) < saveEvery {
		return
	}
	if c.lastNearSave == nil {
		c.lastNearSave = make(map[int]time.Time)
	}
	c.lastNearSave[c.nearCh] = time.Now()
	if b, err := c.exportLocked(); err == nil {
		go writeState(c.nearStatePath(c.nearCh), b, attDb)
	}
}

func writeState(path string, b []byte, attDb float64) {
	// A file younger than a day is left alone, which also covers the first
	// window after every boot. A clock still reading 2010 makes a real file
	// look younger than it is, so the rewrite waits for the time sync rather
	// than happening early.
	if fi, err := os.Stat(path); err == nil {
		if age := time.Since(fi.ModTime()); age >= 0 && age < saveEvery {
			return
		}
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		log.Printf("[aec] saving the echo path: %v", err)
		return
	}
	if err := os.Rename(tmp, path); err != nil {
		log.Printf("[aec] saving the echo path: %v", err)
		return
	}
	log.Printf("[aec] saved the echo path (%.1fdB cancellation, %d bytes)", attDb, len(b))
}

// DefaultStatePath is where the device keeps the saved echo path, beside
// the other files that outlive a process and an OTA.
const DefaultStatePath = "/data/local/etc/echomuse/aec_echo_path.bin"
