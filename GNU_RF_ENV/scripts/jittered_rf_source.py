import numpy as np
from gnuradio import gr, analog, blocks, channels
from pathlib import Path


# ============================================================
# Jittered RF Pulse Source
# ============================================================

class JitteredPulseSource(gr.sync_block):
    """
    Generates a complex pulse envelope with PRI jitter.

    Output:
        amplitude + 0j during the pulse
        0 + 0j outside the pulse
    """

    def __init__(
        self,
        samp_rate=2e6,
        pulse_width_s=10e-6,
        pri_s=100e-6,
        jitter_fraction=0.01,
        seed=42,
        amplitude=1.0,
    ):
        gr.sync_block.__init__(
            self,
            name="Jittered Pulse Source",
            in_sig=None,
            out_sig=[np.complex64],
        )

        if samp_rate <= 0:
            raise ValueError("samp_rate must be > 0")

        if pulse_width_s <= 0:
            raise ValueError("pulse_width_s must be > 0")

        if pri_s <= 0:
            raise ValueError("pri_s must be > 0")

        if pulse_width_s >= pri_s:
            raise ValueError(
                "pulse_width_s must be smaller than pri_s"
            )

        if jitter_fraction < 0:
            raise ValueError(
                "jitter_fraction must be >= 0"
            )

        if amplitude < 0:
            raise ValueError(
                "amplitude must be >= 0"
            )

        self.samp_rate = float(samp_rate)
        self.pulse_width_s = float(pulse_width_s)
        self.pri_s = float(pri_s)
        self.jitter_fraction = float(jitter_fraction)
        self.amplitude = np.float32(amplitude)

        self.rng = np.random.default_rng(seed)

        self.pulse_samples = max(
            1,
            int(round(
                self.pulse_width_s * self.samp_rate
            )),
        )

        self.nominal_pri_samples = max(
            self.pulse_samples + 1,
            int(round(
                self.pri_s * self.samp_rate
            )),
        )

        # First pulse starts immediately.
        self.samples_until_pulse = 0
        self.samples_remaining_in_pulse = 0

    def _next_pri_samples(self):
        """
        Generate the next PRI using uniform fractional jitter.
        """

        jitter = self.rng.uniform(
            -self.jitter_fraction,
            self.jitter_fraction,
        )

        actual_pri_s = self.pri_s * (1.0 + jitter)

        pri_samples = int(
            round(
                actual_pri_s * self.samp_rate
            )
        )

        return max(
            self.pulse_samples + 1,
            pri_samples,
        )

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        i = 0

        while i < n:

            # --------------------------------------------
            # Currently inside a pulse
            # --------------------------------------------

            if self.samples_remaining_in_pulse > 0:

                count = min(
                    self.samples_remaining_in_pulse,
                    n - i,
                )

                out[i:i + count] = (
                    self.amplitude + 0j
                )

                self.samples_remaining_in_pulse -= count
                i += count

                continue

            # --------------------------------------------
            # Waiting for next pulse
            # --------------------------------------------

            if self.samples_until_pulse > 0:

                count = min(
                    self.samples_until_pulse,
                    n - i,
                )

                out[i:i + count] = 0.0 + 0j

                self.samples_until_pulse -= count
                i += count

                continue

            # --------------------------------------------
            # Start a new pulse
            # --------------------------------------------

            out[i] = self.amplitude + 0j

            self.samples_remaining_in_pulse = (
                self.pulse_samples - 1
            )

            # Schedule next pulse using jittered PRI.
            next_pri_samples = self._next_pri_samples()

            self.samples_until_pulse = (
                next_pri_samples
                - self.pulse_samples
            )

            i += 1

        return n


# ============================================================
# Two-Emitter RF Environment
# ============================================================

class TwoEmitterRFSource(gr.top_block):

    def __init__(self):

        gr.top_block.__init__(
            self,
            "Two Emitter RF Environment",
        )

        # ====================================================
        # Global simulation parameters
        # ====================================================

        self.samp_rate = 2e6

        # Finite recording duration.
        self.duration_s = 1.0

        self.total_samples = int(
            self.samp_rate * self.duration_s
        )

        # ====================================================
        # Shared RF noise
        # ====================================================

        # Increase this value to raise the noise floor.
        self.noise_amplitude = 0.05

        # ====================================================
        # Channel impairment parameters
        # ====================================================

        # Normalized frequency offset relative to sample rate.
        #
        # 0.00005 × 2,000,000 = 100 Hz.
        self.channel_frequency_offset = 0.00005

        # Timing scale factor.
        #
        # 1.0 = no timing-rate mismatch.
        self.channel_timing_offset = 1.0

        # ----------------------------------------------------
        # Two-path channel
        # ----------------------------------------------------
        #
        # Path 1:
        #   gain = 0.8
        #   delay = 0 samples
        #
        # Path 2:
        #   gain = 0.2
        #   delay = 1 sample
        #
        # At 2 MS/s:
        #   1 sample = 0.5 us
        #

        self.channel_taps = [
            0.8 + 0j,
            0.2 + 0j,
        ]

        self.channel_seed = 123

        # ====================================================
        # Emitter 1 parameters
        # ====================================================

        self.emitter1_id = 1

        # Logical RF frequency.
        self.emitter1_rf_freq = 3.2e9

        # Local/baseband representation.
        self.emitter1_baseband_freq = 100e3

        self.emitter1_pulse_width_s = 10e-6
        self.emitter1_pri_s = 100e-6
        self.emitter1_jitter_fraction = 0.01
        self.emitter1_amplitude = 1.0
        self.emitter1_seed = 42

        # ====================================================
        # Emitter 2 parameters
        # ====================================================

        self.emitter2_id = 2

        # Logical RF frequency.
        self.emitter2_rf_freq = 8.0e9

        # Local/baseband representation.
        self.emitter2_baseband_freq = -250e3

        self.emitter2_pulse_width_s = 4e-6
        self.emitter2_pri_s = 70e-6
        self.emitter2_jitter_fraction = 0.015
        self.emitter2_amplitude = 0.7
        self.emitter2_seed = 84

        # ====================================================
        # Emitter 1 pulse generator
        # ====================================================

        self.emitter1_pulse = JitteredPulseSource(
            samp_rate=self.samp_rate,
            pulse_width_s=self.emitter1_pulse_width_s,
            pri_s=self.emitter1_pri_s,
            jitter_fraction=self.emitter1_jitter_fraction,
            seed=self.emitter1_seed,
            amplitude=self.emitter1_amplitude,
        )

        # ====================================================
        # Emitter 1 carrier
        # ====================================================

        self.emitter1_carrier = analog.sig_source_c(
            self.samp_rate,
            analog.GR_COS_WAVE,
            self.emitter1_baseband_freq,
            1.0,
            0.0,
        )

        # ====================================================
        # Emitter 1 modulation
        # ====================================================

        self.emitter1_multiply = blocks.multiply_vcc()

        # ====================================================
        # Emitter 2 pulse generator
        # ====================================================

        self.emitter2_pulse = JitteredPulseSource(
            samp_rate=self.samp_rate,
            pulse_width_s=self.emitter2_pulse_width_s,
            pri_s=self.emitter2_pri_s,
            jitter_fraction=self.emitter2_jitter_fraction,
            seed=self.emitter2_seed,
            amplitude=self.emitter2_amplitude,
        )

        # ====================================================
        # Emitter 2 carrier
        # ====================================================

        self.emitter2_carrier = analog.sig_source_c(
            self.samp_rate,
            analog.GR_COS_WAVE,
            self.emitter2_baseband_freq,
            1.0,
            0.0,
        )

        # ====================================================
        # Emitter 2 modulation
        # ====================================================

        self.emitter2_multiply = blocks.multiply_vcc()

        # ====================================================
        # Combine emitters
        # ====================================================

        self.emitter_adder = blocks.add_vcc()

        # ====================================================
        # Shared Gaussian noise
        # ====================================================

        self.noise_source = analog.noise_source_c(
            analog.GR_GAUSSIAN,
            self.noise_amplitude,
            42,
        )

        # ====================================================
        # Add noise to combined signal
        # ====================================================

        self.channel_adder = blocks.add_vcc()

        # ====================================================
        # Multipath / RF channel model
        # ====================================================

        self.channel_model = channels.channel_model(
            noise_voltage=0.0,
            frequency_offset=self.channel_frequency_offset,
            epsilon=self.channel_timing_offset,
            taps=self.channel_taps,
            noise_seed=self.channel_seed,
        )

        # ====================================================
        # Finite recording limiter
        # ====================================================

        self.sample_limit = blocks.head(
            gr.sizeof_gr_complex,
            self.total_samples,
        )

        # ====================================================
        # Output IQ recording
        # ====================================================

        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex,
            str(Path(__file__).resolve().parents[1] / "recordings" / "two_emitter_rf_multipath.dat"),
            False,
        )

        # ====================================================
        # Emitter 1 connections
        # ====================================================

        self.connect(
            self.emitter1_pulse,
            self.emitter1_multiply,
        )

        self.connect(
            self.emitter1_carrier,
            (self.emitter1_multiply, 1),
        )

        # ====================================================
        # Emitter 2 connections
        # ====================================================

        self.connect(
            self.emitter2_pulse,
            self.emitter2_multiply,
        )

        self.connect(
            self.emitter2_carrier,
            (self.emitter2_multiply, 1),
        )

        # ====================================================
        # Combine emitters
        # ====================================================

        self.connect(
            self.emitter1_multiply,
            (self.emitter_adder, 0),
        )

        self.connect(
            self.emitter2_multiply,
            (self.emitter_adder, 1),
        )

        # ====================================================
        # Add shared noise
        # ====================================================

        self.connect(
            self.emitter_adder,
            (self.channel_adder, 0),
        )

        self.connect(
            self.noise_source,
            (self.channel_adder, 1),
        )

        # ====================================================
        # Apply channel / multipath
        # ====================================================

        self.connect(
            self.channel_adder,
            self.channel_model,
        )

        # ====================================================
        # Limit and record
        # ====================================================

        self.connect(
            self.channel_model,
            self.sample_limit,
            self.file_sink,
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    tb = TwoEmitterRFSource()

    print("================================================")
    print(" GNU Radio Two-Emitter RF Environment")
    print("================================================")
    print()

    # ========================================================
    # Global
    # ========================================================

    print("GLOBAL")
    print(
        f"Sample rate       : "
        f"{tb.samp_rate:.0f} S/s"
    )
    print(
        f"Duration          : "
        f"{tb.duration_s:.3f} s"
    )
    print(
        f"Total samples     : "
        f"{tb.total_samples:,}"
    )
    print(
        f"Noise amplitude   : "
        f"{tb.noise_amplitude:.3f}"
    )
    print()

    # ========================================================
    # Channel
    # ========================================================

    print("CHANNEL")

    print(
        f"Frequency offset  : "
        f"{tb.channel_frequency_offset:.8f} normalized"
    )

    print(
        f"Actual freq off   : "
        f"{tb.channel_frequency_offset * tb.samp_rate:.1f} Hz"
    )

    print(
        f"Timing offset     : "
        f"{tb.channel_timing_offset:.6f}"
    )

    print(
        f"Channel taps      : "
        f"{tb.channel_taps}"
    )

    print(
        f"Direct path gain  : "
        f"{abs(tb.channel_taps[0]):.3f}"
    )

    print(
        f"Delayed path gain : "
        f"{abs(tb.channel_taps[1]):.3f}"
    )

    print(
        f"Path delay        : "
        f"0.5 us per additional tap"
    )

    print()

    # ========================================================
    # Emitter 1
    # ========================================================

    print("EMITTER 1")

    print(
        f"Emitter ID        : "
        f"{tb.emitter1_id}"
    )

    print(
        f"Logical RF freq   : "
        f"{tb.emitter1_rf_freq / 1e9:.3f} GHz"
    )

    print(
        f"Baseband offset   : "
        f"{tb.emitter1_baseband_freq / 1e3:.1f} kHz"
    )

    print(
        f"Pulse width       : "
        f"{tb.emitter1_pulse_width_s * 1e6:.3f} us"
    )

    print(
        f"Nominal PRI       : "
        f"{tb.emitter1_pri_s * 1e6:.3f} us"
    )

    print(
        f"PRI jitter        : "
        f"±{tb.emitter1_jitter_fraction * 100:.2f}%"
    )

    print(
        f"Amplitude         : "
        f"{tb.emitter1_amplitude:.3f}"
    )

    print()

    # ========================================================
    # Emitter 2
    # ========================================================

    print("EMITTER 2")

    print(
        f"Emitter ID        : "
        f"{tb.emitter2_id}"
    )

    print(
        f"Logical RF freq   : "
        f"{tb.emitter2_rf_freq / 1e9:.3f} GHz"
    )

    print(
        f"Baseband offset   : "
        f"{tb.emitter2_baseband_freq / 1e3:.1f} kHz"
    )

    print(
        f"Pulse width       : "
        f"{tb.emitter2_pulse_width_s * 1e6:.3f} us"
    )

    print(
        f"Nominal PRI       : "
        f"{tb.emitter2_pri_s * 1e6:.3f} us"
    )

    print(
        f"PRI jitter        : "
        f"±{tb.emitter2_jitter_fraction * 100:.2f}%"
    )

    print(
        f"Amplitude         : "
        f"{tb.emitter2_amplitude:.3f}"
    )

    print()

    # ========================================================
    # Start
    # ========================================================

    print("Starting GNU Radio...")
    print()

    tb.start()

    try:
        tb.wait()

    except KeyboardInterrupt:

        print("Interrupted.")

        tb.stop()
        tb.wait()

    print()
    print(
        "Two-emitter + noise + multipath "
        "flowgraph finished."
    )