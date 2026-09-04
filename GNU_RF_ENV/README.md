# GNU_RF_ENV — GNU Radio RF Environment

## Purpose

Live GNU Radio RF simulation workspace for the SIH 2026 Cognitive EW Smart Scan
project. Generates a continuous two-emitter RF environment with jitter, noise,
channel effects, and multipath, and publishes complex IQ samples via ZMQ for
consumption by a future IQ-to-PDW adapter.

This subsystem lives **inside the master repository** at
`SIH2026_Try2/GNU_RF_ENV`. It sits alongside (NOT inside) the master
application `SIH2026_Try2/cognitive_ew_smart_scan`. The two share one
repository root; GNU RF is an RF-simulation/adapter subsystem that consumes
the authoritative `SieveReceiver` from the master repository.

## Repository Layout (single root)

```
SIH2026_Try2/
├── cognitive_ew_smart_scan/       <- MASTER application (receiver, perception, scheduler)
└── GNU_RF_ENV/                    <- RF simulation + PDW adapter (THIS subsystem)
```

## Python Environments

- **MASTER** (`cognitive_ew_smart_scan`): runs in its own venv / project
  environment (see `cognitive_ew_smart_scan/requirements.txt`).
- **RF** (`GNU_RF_ENV`): requires **RadioConda + GNU Radio** (see below).
  It is run with the RadioConda interpreter because the PDW detector and
  live flowgraphs need `numpy` / GNU Radio.

## GNU Radio Version

- **GNU Radio:** 3.10.12.0
- **Python executable:** `C:\Users\Indrani\radioconda\python.exe`
- (Exact path is machine-specific; the RF tests resolve their paths relative
  to the repository root, so they work from any checkout location.)

## Directory Structure

```
GNU_RF_ENV/
├── flowgraphs/                    # GRC flowgraphs (.grc / generated .py)
│   ├── rf_environment_live.grc    # Live emitter flowgraph (ZMQ PUB)
│   ├── rf_environment_viewer.grc  # Viewer flowgraph (ZMQ SUB + sinks)
│   └── ...                        # step0x single-tone / jittered flowgraphs
├── scripts/
│   ├── live_rf_environment.py     # Live GNU Radio RF source (ZMQ PUB)
│   ├── jittered_rf_source.py      # Jittered-emitter flowgraph module
│   ├── iq_to_pdw.py               # Phase 3A: IQ -> PDW detector
│   ├── frequency_context.py       # Phase 3B: local kHz -> RF MHz mapping
│   ├── iq_bridge.py               # Phase 3C: PDW -> receiver pulse
│   ├── dwell_orchestrator.py      # Phase 3D: per-tune dwell orchestration
│   ├── proof_phase3c.py / proof_phase3d.py   # manual end-to-end proofs
├── tests/                         # Phase 3A-3D + integration unit tests
├── scenarios/                     # (future: scenario definition files)
├── recordings/                    # local IQ recordings (gitignored)
├── metadata/                      # (future: emitter/scenario metadata)
├── .gitignore
└── README.md                      # This file
```

## Architecture — Two-Source Receiver Model

The Cognitive EW Smart Scan project uses a single `SieveReceiver` with two
upstream signal sources:

```
SOURCE A: Existing structured RF world
    │
    │ PulseRecord / environment events
    ▼
    Existing SieveReceiver (cognitive_ew_smart_scan/src/receiver/)


SOURCE B: GNU Radio RF simulation
    │
    │ complex IQ over ZMQ (tcp://127.0.0.1:55555)
    ▼
    IQ-to-PDW adapter (scripts/iq_to_pdw.py + frequency_context + iq_bridge)
    │
    │ receiver-compatible pulse records
    ▼
    Existing SieveReceiver (cognitive_ew_smart_scan/src/receiver/)
```

Both paths converge into ONE Receiver. There must never be two SieveReceiver
implementations.

## Live RF Source

The primary live source is `scripts/live_rf_environment.py`, which generates
a continuous GNU Radio stream and publishes complex IQ samples via ZMQ.

### ZMQ Publisher

- **Endpoint:** `tcp://*:55555`
- **Format:** complex64 samples
- **Sample rate:** 2 MS/s

### GRC Viewer Subscriber

The GRC viewer flowgraph (`grc/rf_environment_viewer.grc`) subscribes to:

- **Endpoint:** `tcp://127.0.0.1:55555`
- **Sinks:** Time Sink, Frequency Sink, Waterfall Sink

## Validated RF Parameters

### Emitter 1

| Parameter | Value |
|---|---|
| Logical RF frequency | 3.2 GHz |
| Baseband (sampled) frequency | +100 kHz |
| Pulse width | 10 µs |
| Nominal PRI | 100 µs |
| PRI jitter | ±1% |
| Amplitude | 1.0 |
| RNG seed | 42 |

### Emitter 2

| Parameter | Value |
|---|---|
| Logical RF frequency | 8.0 GHz |
| Baseband (sampled) frequency | -250 kHz |
| Pulse width | 4 µs |
| Nominal PRI | 70 µs |
| PRI jitter | ±1.5% |
| Amplitude | 0.7 |
| RNG seed | 84 |

### Shared Noise

- Gaussian noise amplitude: 0.05

### Channel Model

| Parameter | Value |
|---|---|
| Normalized frequency offset | 0.00005 |
| Actual frequency offset at 2 MS/s | 100 Hz |
| Timing scale | 1.0 |
| Taps | `[0.8+0j, 0.2+0j]` |
| Second tap delay | 1 sample = 0.5 µs at 2 MS/s |

### Sample Rate

- 2 MS/s

## Important RF Modeling Note

The 3.2 GHz and 8.0 GHz values are **logical RF frequencies** for scenario
documentation. The actual sampled GNU Radio representation is baseband:
+100 kHz and -250 kHz at 2 MS/s. Do not describe the current simulation as
directly sampling a 3.2 GHz or 8 GHz carrier.

## Recordings

The `recordings/` directory contains local development IQ recordings (gitignored):

| File | Purpose |
|---|---|
| `jittered_rf_complex.dat` | Single emitter with jitter (dev/test) |
| `two_emitter_rf_complex.dat` | Two emitter raw complex IQ |
| `two_emitter_rf_with_noise.dat` | Two emitter + Gaussian noise |
| `two_emitter_rf_channel.dat` | Two emitter + channel model |
| `two_emitter_rf_multipath.dat` | Two emitter + multipath (referenced by `rf_environment_viewer.grc`) |

All recordings are ~15 MB each (20M complex64 samples at 2 MS/s). These are
local development artifacts, not committed to version control.

## IQ-to-PDW Adapter (Implemented, Phases 3A-3F)

The integration boundary is implemented and tested:

```
GNU Radio IQ (ZMQ complex64)
    │
    ▼
scripts/iq_to_pdw.py  (Phase 3A)   IQ pulse detection / measurement -> PDW
    │
    ▼
scripts/frequency_context.py (Phase 3B)  local kHz -> logical RF MHz
    │
    ▼
scripts/iq_bridge.py     (Phase 3C)      PDW -> receiver pulse record
    │
    ▼
scripts/dwell_orchestrator.py (Phase 3D) per-tune dwell orchestration
    │
    ▼
existing SieveReceiver.add_pulse()  (authoritative, located in the MASTER repo)
```

The adapter estimates observable quantities (ToA, local frequency, PW,
amplitude). AoA is NOT fabricated from a single IQ stream
(`aoa_deg = 0.0` placeholder). Ground-truth emitter IDs / true RF / true
PW / true PRI are NOT injected into the observable receiver path.

`GNU_RF_ENV` does NOT contain a duplicate receiver. It consumes the
authoritative `SieveReceiver` from `cognitive_ew_smart_scan/src/receiver/`
through a checkout-relative integration path
(`repo_root/cognitive_ew_smart_scan/src`). There must never be a second
`SieveReceiver` implementation. See `tests/test_repo_integration.py` for the
cross-repository integration proof.

## Running

Start the live RF source:

```bash
C:\Users\Indrani\radioconda\python.exe scripts/live_rf_environment.py
```

Then open the GRC viewer flowgraph to see the signal.
