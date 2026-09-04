"""
Jittered RF Pulse Envelope Generator

Generates a complex-valued pulse envelope:
    1+0j during each pulse
    0+0j otherwise

The pulse width is fixed.
The PRI varies pulse-to-pulse within a bounded jitter range.
"""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """
    Generate a reproducible jittered pulse train.
    """

    def __init__(
        self,
        samp_rate=2e6,
        pulse_width_s=10e-6,
        pri_s=100e-6,
        jitter_fraction=0.01,
        seed=42,
    ):
        gr.sync_block.__init__(
            self,
            name="Jittered Pulse Envelope",
            in_sig=None,
            out_sig=[np.complex64],
        )

        # Store parameters.
        self.samp_rate = float(samp_rate)
        self.pulse_width_s = float(pulse_width_s)
        self.pri_s = float(pri_s)
        self.jitter_fraction = float(jitter_fraction)
        self.seed = int(seed)

        # Validate parameters.
        if self.samp_rate <= 0:
            raise ValueError("samp_rate must be > 0")

        if self.pulse_width_s <= 0:
            raise ValueError("pulse_width_s must be > 0")

        if self.pri_s <= 0:
            raise ValueError("pri_s must be > 0")

        if self.pulse_width_s >= self.pri_s:
            raise ValueError(
                "pulse_width_s must be smaller than pri_s"
            )

        if not 0.0 <= self.jitter_fraction < 1.0:
            raise ValueError(
                "jitter_fraction must be in [0, 1)"
            )

        # Convert pulse width to samples.
        self.pulse_samples = max(
            1,
            int(round(self.pulse_width_s * self.samp_rate)),
        )

        self.rng = np.random.default_rng(self.seed)

        # State of the currently generated pulse train.
        self.samples_until_pulse = 0
        self.samples_remaining_in_pulse = 0

        # Start with a nominal first pulse at t = 0.
        self.first_pulse = True

    def _next_pri_samples(self):
        """
        Generate the next PRI using bounded uniform jitter.
        """
        delta = self.rng.uniform(
            -self.jitter_fraction,
            self.jitter_fraction,
        )

        actual_pri_s = self.pri_s * (1.0 + delta)

        return max(
            self.pulse_samples + 1,
            int(round(actual_pri_s * self.samp_rate)),
        )

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        i = 0

        while i < n:

            # Start the first pulse immediately.
            if self.first_pulse:
                self.samples_remaining_in_pulse = self.pulse_samples
                self.first_pulse = False

            # Generate the active pulse.
            if self.samples_remaining_in_pulse > 0:
                count = min(
                    self.samples_remaining_in_pulse,
                    n - i,
                )

                out[i:i + count] = np.complex64(1.0 + 0.0j)

                self.samples_remaining_in_pulse -= count
                i += count

                # If this pulse ended, schedule the next pulse.
                if self.samples_remaining_in_pulse == 0:
                    next_pri = self._next_pri_samples()

                    self.samples_until_pulse = (
                        next_pri - self.pulse_samples
                    )

                continue

            # Generate the silent section between pulses.
            if self.samples_until_pulse > 0:
                count = min(
                    self.samples_until_pulse,
                    n - i,
                )

                out[i:i + count] = np.complex64(0.0 + 0.0j)

                self.samples_until_pulse -= count
                i += count

                continue

            # Safety fallback.
            self.samples_remaining_in_pulse = self.pulse_samples

        return n