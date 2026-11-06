package aec

import (
	"math"
	"os"
	"testing"
	"time"
)

// echoOn returns the near end for signal through one path: delay samples,
// linear gain, polarity.
func echoOn(signal []int16, delay int, gain float64, invert bool) []int16 {
	mic := make([]int16, len(signal))
	for i := delay; i < len(signal); i++ {
		v := float64(signal[i-delay]) * gain
		if invert {
			v = -v
		}
		mic[i] = int16(v)
	}
	return mic
}

// attenuationOver runs frames [from, from+n) of signal and its echo through
// the canceller and returns the cancellation over them, in dB.
func attenuationOver(c *Canceller, signal, mic []int16, from, n int) float64 {
	var in, out float64
	for f := from; f < from+n; f++ {
		lo, hi := f*FrameSize, (f+1)*FrameSize
		mb := toBytes(mic[lo:hi])
		res := c.ProcessWithRef(mb, toBytes(signal[lo:hi]))
		in += rms(mb)
		out += rms(res)
	}
	return 20 * math.Log10(in/out)
}

// The bench of 2026-09-24: the beam lock moved the near end from one mic to
// another between sessions, and the filter converged on the first cancelled
// 1.2 dB on the second. With a state parked per channel, a mic that comes
// back cancels from its first frames.
func TestNearStateRestoredWhenAMicReturns(t *testing.T) {
	signal := synth(400 * FrameSize)
	pathA := echoOn(signal, 33, 1.0, true)   // the centre mic
	pathB := echoOn(signal, 41, 0.45, false) // a perimeter mic on the far side

	c := hwCanceller(64)
	c.SelectNear(6)
	attenuationOver(c, signal, pathA, 0, 150) // converge on A
	c.SelectNear(2)
	onB := attenuationOver(c, signal, pathB, 150, 150) // converge on B
	c.SelectNear(6)
	back := attenuationOver(c, signal, pathA, 300, 10)

	control := hwCanceller(64)
	attenuationOver(control, signal, pathA, 0, 150)
	attenuationOver(control, signal, pathB, 150, 150) // same history, one filter
	cold := attenuationOver(control, signal, pathA, 300, 10)

	t.Logf("on B %.1fdB; first 10 frames back on A: parked %.1fdB, one filter %.1fdB", onB, back, cold)
	if back < cold+10 {
		t.Fatalf("parked state did not help: %.1fdB against %.1fdB with one filter", back, cold)
	}
	if back < 12 {
		t.Fatalf("restored filter cancels only %.1fdB", back)
	}
}

// A channel never seen before carries on with the current filter rather
// than an empty one: the loudspeaker, the converters and their delay are
// shared, so it starts closer than nothing.
func TestNearStateUnseenMicCarriesTheCurrentPath(t *testing.T) {
	signal := synth(200 * FrameSize)
	pathA := echoOn(signal, 33, 1.0, true)
	c := hwCanceller(64)
	c.SelectNear(6)
	attenuationOver(c, signal, pathA, 0, 150)
	c.SelectNear(3) // never seen
	carried := attenuationOver(c, signal, pathA, 150, 10)
	if carried < 12 {
		t.Fatalf("moving to an unseen mic lost the filter: %.1fdB on the same path", carried)
	}
	if _, parked := c.nearStates[6]; !parked {
		t.Fatal("the centre mic's path was not parked")
	}
}

func TestNearStateSameChannelAndRebuild(t *testing.T) {
	c := hwCanceller(300)
	c.SelectNear(6)
	c.SelectNear(6)
	if len(c.nearStates) != 0 {
		t.Fatal("a repeated channel parked a state")
	}
	c.SelectNear(1)
	if len(c.nearStates) != 1 {
		t.Fatalf("parked %d states, want 1", len(c.nearStates))
	}
	c.SetHardwareRef(false) // rebuilds the filter at the software tap's 300ms
	if len(c.nearStates) != 0 {
		t.Fatal("parked states survived a rebuild they cannot load into")
	}
	if c.NearChannel() != 1 {
		t.Fatalf("near channel %d, want 1", c.NearChannel())
	}
}

// Before anything is learnt (cancellation off, or the first channel told)
// there is nothing to park and nothing to say.
func TestNearStateBeforeAnyFilter(t *testing.T) {
	c := New()
	c.SelectNear(6)
	c.SelectNear(2)
	if len(c.nearStates) != 0 || c.NearChannel() != 2 {
		t.Fatalf("disabled canceller parked %d states, channel %d", len(c.nearStates), c.NearChannel())
	}
}

// The parked paths outlive a restart: each mic's path is saved the first
// time it converges and loaded into its slot when the hardware filter is
// built, so a mic the lock picks after a reboot cancels from its first frames.
func TestPerMicPathsSurviveARestart(t *testing.T) {
	path := t.TempDir() + "/aec_echo_path.bin"
	signal := synth(400 * FrameSize)
	pathA := echoOn(signal, 33, 1.0, true)
	pathB := echoOn(signal, 41, 0.45, false)

	first := New()
	first.SetStatePath(path)
	first.SetParams(true, 0, 300)
	first.SetHardwareRef(true)
	first.SelectNear(6)
	attenuationOver(first, signal, pathA, 0, 150)
	first.SelectNear(2)
	attenuationOver(first, signal, pathB, 150, 150) // the telemetry saves ch2's file once it converges
	var saved []byte
	for i := 0; i < 400 && saved == nil; i++ {
		saved, _ = os.ReadFile(path + ".ch2")
		time.Sleep(5 * time.Millisecond)
	}
	if saved == nil {
		t.Fatal("ch2's converged path was not saved")
	}

	restarted := New()
	restarted.SetStatePath(path)
	restarted.SetParams(true, 0, 300)
	restarted.SetHardwareRef(true) // loads the files
	if _, ok := restarted.nearStates[2]; !ok {
		t.Fatal("ch2's saved path was not loaded into its slot")
	}
	restarted.SelectNear(6)
	restarted.SelectNear(2)
	back := attenuationOver(restarted, signal, pathB, 300, 10)
	cold := attenuationOver(hwCanceller(300), signal, pathB, 300, 10)
	t.Logf("first 10 frames on B after a restart: %.1fdB, cold %.1fdB", back, cold)
	if back < cold+10 || back < 12 {
		t.Fatalf("the saved per-mic path did not help: %.1fdB against %.1fdB cold", back, cold)
	}
}
