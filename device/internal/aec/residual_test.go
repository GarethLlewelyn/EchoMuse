package aec

import (
	"math"
	"testing"
)

// bursty gates a signal like speech: 240 ms on, 160 ms off, so the
// preprocessor's noise tracker sees pauses the way it does in a room. A
// steady far end would let it learn the residual as noise, which the 0 dB
// noise floor then protects on purpose.
func bursty(signal []int16) []int16 {
	out := make([]int16, len(signal))
	period := 400 * sampleRate / 1000
	on := 240 * sampleRate / 1000
	for i, v := range signal {
		if i%period < on {
			out[i] = v
		}
	}
	return out
}

// What the stage is for: the residual the linear filter leaves while it is
// short of the path, which on the bench is the first second of a reply after
// the beam moved the mic or the volume moved the level. That residual is the
// filter's own estimate scaled, exactly what its leak estimate sees, so the
// stage takes it down while the filter catches up. Here the path's gain
// doubles after convergence and the second after the change is measured.
// (The opposite step, a path that shrinks, leaves an anti-correlated residual
// that speex's leak estimate reads as none; the filter alone covers that.)
func TestResidualSuppressionCoversTheFilterCatchingUp(t *testing.T) {
	signal := bursty(synthAt(400*FrameSize, 3000))
	before := echoOn(signal, 33, 0.5, true)
	after := echoOn(signal, 33, 1.0, true)

	plain := hwCanceller(64)
	plain.SetResidualSuppression(0)
	attenuationOver(plain, signal, before, 0, 200)
	linear := attenuationOver(plain, signal, after, 200, 30)

	deep := hwCanceller(64) // DefaultResidualDb
	if deep.ResidualSuppression() != DefaultResidualDb {
		t.Fatalf("default depth %d, want %d", deep.ResidualSuppression(), DefaultResidualDb)
	}
	attenuationOver(deep, signal, before, 0, 200)
	suppressed := attenuationOver(deep, signal, after, 200, 30)

	t.Logf("the second after the path's gain doubled: linear %.1fdB, with residual suppression %.1fdB", linear, suppressed)
	// Measured at +4.6 dB (17.6 to 22.1) on 2026-09-24; the field case, a
	// residual of speech through a real path, is what the stage is judged on.
	if suppressed < linear+3 {
		t.Fatalf("suppression added only %.1fdB over the linear filter's %.1fdB", suppressed-linear, linear)
	}
}

// With nothing playing there is no residual, and the person's voice must
// come through at its level: noise suppression sits at 0 dB.
func TestResidualSuppressionLeavesTheNearEndAlone(t *testing.T) {
	c := hwCanceller(64)
	voice := bursty(synthAt(120*FrameSize, 3000))
	quiet := make([]int16, len(voice))
	var in, out float64
	for f := 0; f < 120; f++ {
		lo, hi := f*FrameSize, (f+1)*FrameSize
		mb := toBytes(voice[lo:hi])
		res := c.ProcessWithRef(mb, toBytes(quiet[lo:hi]))
		if f >= 40 { // past the preprocessor's warm-up
			in += rms(mb)
			out += rms(res)
		}
	}
	change := 20 * math.Log10(out/in)
	t.Logf("near end with no far end: %.2fdB", change)
	if change < -1.5 || change > 1.5 {
		t.Fatalf("the suppression stage changed the person's level by %.1fdB with nothing playing", change)
	}
}

// Off means off: the stage is dropped and the frame is the filter's alone.
func TestResidualSuppressionCanBeSwitchedOff(t *testing.T) {
	c := hwCanceller(64)
	c.SetResidualSuppression(0)
	if c.pp != nil || c.ResidualSuppression() != 0 {
		t.Fatal("stage still built with suppression off")
	}
	c.SetResidualSuppression(30)
	if c.pp == nil || c.ResidualSuppression() != 30 {
		t.Fatal("stage not rebuilt at 30dB")
	}
	c.SetHardwareRef(false) // rebuilds the filter: the stage must follow it
	if c.pp == nil {
		t.Fatal("stage lost across a filter rebuild")
	}
}
