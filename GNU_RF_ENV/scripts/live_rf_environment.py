import numpy as np

from gnuradio import gr, analog, blocks, channels, zeromq


# ============================================================
# Jittered Pulse Source
# ============================================================

class JitteredPulseSource(gr.sync_block):
    """
    Infinite complex pulse generator with independent PRI jitter.
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

        if pri_s <= pulse_width_s:
            raise ValueError(
                "pri_s must be greater than pulse_width_s"
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

        self.samples_until_pulse = 0
        self.samples_remaining_in_pulse = 0

    def _next_pri_samples(self):
        jitter = self.rng.uniform(
            -self.jitter_fraction,
            self.jitter_fraction,
        )

        actual_pri_s = self.pri_s * (1.0 + jitter)

        return max(
            self.pulse_samples + 1,
            int(round(
                actual_pri_s * self.samp_rate
            )),
        )

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        i = 0

        while i < n:

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

            if self.samples_until_pulse > 0:

                count = min(
                    self.samples_until_pulse,
                    n - i,
                )

                out[i:i + count] = 0.0 + 0j

                self.samples_until_pulse -= count
                i += count

                continue

            # Start pulse immediately.
            out[i] = self.amplitude + 0j

            self.samples_remaining_in_pulse = (
                self.pulse_samples - 1
            )

            next_pri_samples = self._next_pri_samples()

            self.samples_until_pulse = (
                next_pri_samples
                - self.pulse_samples
            )

            i += 1

        return n


# ============================================================
# Live RF Environment
# ============================================================

class LiveRFEnvironment(gr.top_block):

    def __init__(self):

        gr.top_block.__init__(
            self,
            "Live RF Environment",
        )

        # ====================================================
        # Global parameters
        # ====================================================

        self.samp_rate = 2e6

        # ====================================================
        # Emitter 1
        # ====================================================

        self.emitter1_id = 1

        self.emitter1_rf_freq = 3.2e9
        self.emitter1_baseband_freq = 100e3

        self.emitter1_pulse_width_s = 10e-6
        self.emitter1_pri_s = 100e-6
        self.emitter1_jitter_fraction = 0.01
        self.emitter1_amplitude = 1.0
        self.emitter1_seed = 42

        # ====================================================
        # Emitter 2
        # ====================================================

        self.emitter2_id = 2

        self.emitter2_rf_freq = 8.0e9
        self.emitter2_baseband_freq = -250e3

        self.emitter2_pulse_width_s = 4e-6
        self.emitter2_pri_s = 70e-6
        self.emitter2_jitter_fraction = 0.015
        self.emitter2_amplitude = 0.7
        self.emitter2_seed = 84

        # ====================================================
        # Shared noise
        # ====================================================

        self.noise_amplitude = 0.05
        self.noise_seed = 42

        # ====================================================
        # Channel
        # ====================================================

        self.channel_frequency_offset = 0.00005
        self.channel_timing_offset = 1.0

        self.channel_taps = [
            0.8 + 0j,
            0.2 + 0j,
        ]

        self.channel_seed = 123

        # ====================================================
        # Emitter 1 pulse source
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

        self.emitter1_multiply = blocks.multiply_vcc()

        # ====================================================
        # Emitter 2 pulse source
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
            self.noise_seed,
        )

        self.noise_adder = blocks.add_vcc()

        # ====================================================
        # Channel
        # ====================================================

        self.channel_model = channels.channel_model(
            noise_voltage=0.0,
            frequency_offset=self.channel_frequency_offset,
            epsilon=self.channel_timing_offset,
            taps=self.channel_taps,
            noise_seed=self.channel_seed,
        )

        # ====================================================
        # ZMQ PUB
        # ====================================================
        #
        # Publishes complex64 samples to:
        #
        # tcp://*:55555
        #
        # GRC subscriber connects to:
        #
        # tcp://127.0.0.1:55555
        #

        self.zmq_pub = zeromq.pub_sink(
            gr.sizeof_gr_complex,
            1,
            "tcp://*:55555",
            100,
            False,
            -1,
        )

        # ====================================================
        # Connections
        # ====================================================

        # Emitter 1
        self.connect(
            self.emitter1_pulse,
            self.emitter1_multiply,
        )

        self.connect(
            self.emitter1_carrier,
            (self.emitter1_multiply, 1),
        )

        # Emitter 2
        self.connect(
            self.emitter2_pulse,
            self.emitter2_multiply,
        )

        self.connect(
            self.emitter2_carrier,
            (self.emitter2_multiply, 1),
        )

        # Combine
        self.connect(
            self.emitter1_multiply,
            (self.emitter_adder, 0),
        )

        self.connect(
            self.emitter2_multiply,
            (self.emitter_adder, 1),
        )

        # Noise
        self.connect(
            self.emitter_adder,
            (self.noise_adder, 0),
        )

        self.connect(
            self.noise_source,
            (self.noise_adder, 1),
        )

        # Channel
        self.connect(
            self.noise_adder,
            self.channel_model,
        )

        # Live stream to ZMQ
        self.connect(
            self.channel_model,
            self.zmq_pub,
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    tb = LiveRFEnvironment()

    print("================================================")
    print(" GNU Radio LIVE RF Environment")
    print("================================================")
    print()

    print("GLOBAL")
    print(
        f"Sample rate       : "
        f"{tb.samp_rate:.0f} S/s"
    )
    print()

    print("EMITTER 1")
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
    print()

    print("EMITTER 2")
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
    print()

    print("CHANNEL")
    print(
        f"Noise amplitude   : "
        f"{tb.noise_amplitude:.3f}"
    )
    print(
        f"Frequency offset  : "
        f"{tb.channel_frequency_offset:.8f} normalized"
    )
    print(
        f"Actual freq off   : "
        f"{tb.channel_frequency_offset * tb.samp_rate:.1f} Hz"
    )
    print(
        f"Channel taps      : "
        f"{tb.channel_taps}"
    )
    print(
        "PUB address       : "
        "tcp://*:55555"
    )
    print()

    print("Starting live GNU Radio environment...")
    print("Press Ctrl+C to stop.")
    print()

    tb.start()

    try:
        while True:
            input()
    except KeyboardInterrupt:
        print()
        print("Stopping...")

    tb.stop()
    tb.wait()

    print("Live RF environment stopped.")