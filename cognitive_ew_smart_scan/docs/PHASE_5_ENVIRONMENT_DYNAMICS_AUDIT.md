# PHASE 5A — ENVIRONMENT DYNAMICS & FREQUENCY-AGILITY AUDIT

**Project:** Cognitive EW Smart Scan Scheduler  
**Repository:** `cognitive_ew_smart_scan` (`SIH2026_Try2`)  
**Date:** September 11, 2026  
**Auditors:** ML Systems Engineering Lead  

---

## 1. Executive Summary

Phase 5 addresses the core physical and behavioral challenge of the Smart Scan Scheduler: **interception of frequency-agile and sparse emitters**. 

Following the restoration of true flat 180-action ML control (Phase 3) and reward alignment to dominant physical interception (Phase 4), this audit examines the underlying environment dynamics, receiver temporal limits, observation causality, and the learning mechanics that govern performance against non-stationary emitters.

---

## 2. Answers to the 10 Mandatory Audit Questions

### Q1: How does emitter frequency change over time?
In TSRD recorded scenarios (`data/stare/` and `data/scan/`), pulse records specify explicit carrier frequencies ($f_c$), pulse widths ($PW$), and Times of Arrival ($ToA$). For frequency-agile radars (e.g. `config_119`), pulses hop between discrete frequencies across successive pulse repetitions. In synthetic scenarios, `synthetic_records` creates staggered burst trains where each emitter has an assigned carrier frequency with minor jitter, or discrete band hops.

### Q2: Is frequency hopping actually represented at the receiver?
**Yes.** In `SieveReceiver` ([`src/receiver/sieve_receiver.py`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/cognitive_ew_smart_scan/src/receiver/sieve_receiver.py)), frequency filtering is explicitly physical:
```python
def frequency_in_window(self, frequency_mhz: float) -> bool:
    lower, upper = self.get_frequency_window()  # [center - IBW/2, center + IBW/2]
    return lower <= float(frequency_mhz) <= upper
```
If an emitter hops from $2,750\text{ MHz}$ (Band 5) to $6,250\text{ MHz}$ (Band 12), a receiver tuned to Band 5 ($[2500, 3000]\text{ MHz}$) will observe zero pulses from that emitter during that dwell.

### Q3: Can the receiver only observe the selected band?
**Yes.** The instantaneous receiver bandwidth (IBW) is strictly $500\text{ MHz}$. Pulses occurring outside the tuned window $[f_c - 250, f_c + 250]\text{ MHz}$ are rejected by `_evaluate_interval()`. Only the selected band's pulses enter `observation.detections`.

### Q4: Does emitter movement between bands occur between decisions?
**Yes.** The RF world timeline advances continuously via `RadioEnvironment`'s event queue. Between scheduler decision $t$ and decision $t+1$, real time elapses by the dwell duration (e.g., $250\,\mu\text{s}$, $500\,\mu\text{s}$, or $1000\,\mu\text{s}$). If an agile emitter's hop interval or PRI falls within or between dwells, its frequency shifts while the scheduler is processing or tuning.

### Q5: Is ToA handled causally?
**Yes.** The world is advanced strictly up to `dwell_end` before the dwell execution via `_advance_world_to(dwell_end)`. Only entry events where `toa_us <= dwell_end` enter the receiver pulse buffer. Events with `time > dwell_end` remain queued in `RadioEnvironment` and are invisible to the receiver and scheduler.

### Q6: Does the receiver miss pulses because of dwell timing?
**Yes.** A pulse is only detected if its active interval $[ToA, ToA + PW]$ overlaps the dwell interval $[T_{\text{dwell\_start}}, T_{\text{dwell\_end}}]$ AND its frequency falls within the tuned IBW. If an emitter produces a $5\,\mu\text{s}$ pulse in Band 5 at $t=510\,\mu\text{s}$, but the scheduler dwelt in Band 5 only from $t=[0, 500]\,\mu\text{s}$ and switched to Band 6 at $t=500\,\mu\text{s}$, the pulse is missed physically.

### Q7: Can an emitter hop multiple bands between scheduler decisions?
**Yes.** An agile radar with a fast hop rate (e.g., hop period of $100\,\mu\text{s}$) will hop up to 5 times during a standard $500\,\mu\text{s}$ dwell, potentially traversing multiple 500 MHz bands before the scheduler makes its next decision.

### Q8: Do sparse emitters remain active while the scheduler is scanning elsewhere?
**Yes.** Sparse emitters produce pulses in their respective bands according to their physical ToA schedule regardless of where the scheduler is tuned. These missed opportunities are recorded in `active_bands_vec` for coverage FOM and missed penalty calculations, but never leak into the scheduler's observation.

### Q9: Do episode resets accidentally create artificial stationarity?
**No.** In `train_scheduler.py`, `env.reset()` draws a fresh random TSRD scenario file from the eligible dataset via `records_provider=train_source.sample`. The receiver clock resets to $t=0$, but the pulse sequence is drawn from diverse files with varying emitter populations.

### Q10: Does the environment expose sufficient temporal history for the DRQN to infer agility?
**Partially in Belief, Fully in DRQN recurrent state.**
- The belief state vector (360 floats = 36 bands $\times$ 10 features) maintains:
  - Feature 0: `occupancy_prob` (EMA with $\alpha=0.3$)
  - Feature 2: `revisit_age` (steps since last visit)
  - Feature 4: `uncertainty` (epistemic + activity uncertainty)
  - Feature 7: `periodicity_stability` (PRI consistency from pulse history)
  - Feature 8: `agility_indicator` (frequency dispersion / hopping indicator)
- Furthermore, the 2-layer DRQN maintains an internal LSTM hidden state across a sequence of 16 steps (with 8 steps burn-in), enabling the recurrent network to learn temporal hopping dynamics.

---

## 3. Physical Temporal Resolution & Hop Classification

| Parameter | Nominal Value | Range / Options | Notes |
| :--- | :---: | :---: | :--- |
| **Base Dwell Time** | $500.0\,\mu\text{s}$ | $250 - 1000\,\mu\text{s}$ | Mode 0 (SHORT)=250, Mode 1 (NORMAL)=500, Mode 2 (LONG)=1000 |
| **Pulse Width ($PW$)** | $1.0 - 10.0\,\mu\text{s}$ | $0.2 - 100\,\mu\text{s}$ | Typical radar pulse |
| **PRI ($ToA$ Spacing)** | $200 - 1500\,\mu\text{s}$ | $50 - 5000\,\mu\text{s}$ | Pulse repetition interval |
| **Decision Rate** | $1.0 - 4.0\text{ kHz}$ | $1000 - 4000\text{ decisions/s}$ | Dwell interval |

### Hop Rate to Decision Rate Classification:
$$\rho_{\text{hop}} = \frac{T_{\text{dwell}}}{T_{\text{hop}}}$$
1. **Stationary ($\rho = 0$):** Fixed frequency. Optimal strategy: periodic revisit for track maintenance.
2. **Slow Hopper ($\rho < 0.2$):** Emitter stays in a band for $\ge 5$ dwell decisions ($T_{\text{hop}} \ge 2500\,\mu\text{s}$). Easily tracked by belief EMA.
3. **Medium Hopper ($0.2 \le \rho \le 1.0$):** Emitter changes bands every $1 - 5$ dwells. Requires predictive switching and agility belief.
4. **Fast Hopper ($1.0 < \rho \le 5.0$):** Emitter changes bands within a single dwell. Interception requires wideband coincidence or predictive lead.
5. **Ultra-Fast Hopper ($\rho > 5.0$):** Emitter hops faster than receiver dwell. Interception is probabilistic / stochastic.

---

## 4. 36-Band Discretization & Edge Boundary Verification

The RF spectrum spans $0\text{ MHz}$ to $18,000\text{ MHz}$ across 36 bands:
$$\Delta f = \frac{18000 - 0}{36} = 500.0\text{ MHz/band}$$
$$f_{\text{center}}(b) = 0 + 500 \times (b + 0.5) = 500b + 250\text{ MHz}$$
$$\text{Window}(b) = [500b, 500(b+1)]\text{ MHz}$$

Boundary checks:
- $0.0\text{ MHz} \to \text{Band } 0$ ($[0, 500]\text{ MHz}$)
- $499.999\text{ MHz} \to \text{Band } 0$ ($[0, 500]\text{ MHz}$)
- $500.0\text{ MHz} \to \text{Band } 1$ ($[500, 1000]\text{ MHz}$)
- $17,500.0\text{ MHz} \to \text{Band } 35$ ($[17500, 18000]\text{ MHz}$)
- $18,000.0\text{ MHz} \to \text{Band } 35$ (Clamped to legal maximum)

Zero off-by-one errors, zero dropped edge pulses, and continuous contiguous coverage verified.
