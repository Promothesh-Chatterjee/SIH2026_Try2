---
name: SMARTSCAN EW — Tactical Spectrum Operations
colors:
  surface: '#111317'
  surface-dim: '#111317'
  surface-bright: '#37393e'
  surface-container-lowest: '#0c0e12'
  surface-container-low: '#1a1c20'
  surface-container: '#1e2024'
  surface-container-high: '#282a2e'
  surface-container-highest: '#333539'
  on-surface: '#e2e2e8'
  on-surface-variant: '#c6c5d5'
  inverse-surface: '#e2e2e8'
  inverse-on-surface: '#2f3035'
  outline: '#908f9e'
  outline-variant: '#454653'
  surface-tint: '#bdc2ff'
  primary: '#bdc2ff'
  on-primary: '#0b1c93'
  primary-container: '#0e1e94'
  on-primary-container: '#8391ff'
  inverse-primary: '#4553c1'
  secondary: '#96ccff'
  on-secondary: '#003353'
  secondary-container: '#3097e0'
  on-secondary-container: '#002c48'
  tertiary: '#49df9d'
  on-tertiary: '#003823'
  tertiary-container: '#003a24'
  on-tertiary-container: '#00b075'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dfe0ff'
  primary-fixed-dim: '#bdc2ff'
  on-primary-fixed: '#000a64'
  on-primary-fixed-variant: '#2b39a8'
  secondary-fixed: '#cee5ff'
  secondary-fixed-dim: '#96ccff'
  on-secondary-fixed: '#001d32'
  on-secondary-fixed-variant: '#004a75'
  tertiary-fixed: '#6afcb8'
  tertiary-fixed-dim: '#49df9d'
  on-tertiary-fixed: '#002112'
  on-tertiary-fixed-variant: '#005234'
  background: '#111317'
  on-background: '#e2e2e8'
  surface-variant: '#333539'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.02em
  display-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.01em
  headline-panel:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.08em
  body-default:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
    letterSpacing: 0.01em
  body-bold:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.01em
  telemetry-lg:
    fontFamily: JetBrains Mono
    fontSize: 18px
    fontWeight: '700'
    lineHeight: 22px
    letterSpacing: 0.02em
  telemetry-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0em
  telemetry-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 12px
    letterSpacing: 0.02em
  waterfall-marker:
    fontFamily: JetBrains Mono
    fontSize: 9px
    fontWeight: '400'
    lineHeight: 10px
    letterSpacing: 0.05em
  command-badge:
    fontFamily: JetBrains Mono
    fontSize: 9px
    fontWeight: '700'
    lineHeight: 10px
    letterSpacing: 0.1em
spacing:
  unit-2xs: 0.125rem
  unit-xs: 0.25rem
  unit-sm: 0.375rem
  unit-md: 0.5rem
  unit-lg: 0.75rem
  unit-xl: 1rem
  unit-2xl: 1.5rem
  dock-gap: 0.25rem
  panel-padding-tight: 0.5rem
  panel-padding-standard: 0.75rem
  column-gutter: 0.5rem
---

## Brand & Style

This design system targets signals intelligence (SIGINT), electronic warfare (EW) officers, and mission telemetry directors operating in high-stress, information-dense mission control environments. The emotional posture is calculated, vigilant, non-fatiguing, and relentlessly precise. Operators require instantaneous threat recognition, zero decorative friction, and absolute clarity under varying ambient cockpit or control-room conditions.

The aesthetic fuses **Tactical Technical Realism** and **Military Ergonomics**. It abandons consumer soft-UI conventions in favor of modular telemetry instrumentation, razor-sharp architectural partitions, monospaced tabular telemetry, and disciplined functional signaling. Contrast is strictly reserved for actionable intelligence: nominal conditions remain recessed within deep graphite tiers, while RF emissions, jamming vectors, and classified alerts cut through with calibrated chromatic energy.

## Colors

The palette enforces a strict optical hierarchy calibrated for dark-room command centers:
- **Base Environment**: Deep Void (`#07080B`) and Primary Graphite (`#0B0D11`) form the foundational backdrops to eliminate backlight bleed and eye fatigue over extended operational watches.
- **Structural Elevation**: Modular panels and telemetry docks employ `#12161F` and `#181E2A`, outlined by radar-grid borders (`#1E293B` and `#334155`).
- **Telemetry Navy & Electric Teal (`#0E1E94`, `#1E8DD5`)**: Primary system interactions, cursor Reticles, active RF trace lines, and nominal tracking vectors.
- **Signal Emerald (`#00BB7D`)**: Verified friendly signals (IFF Mode 5), locked RF channels, validated decryptions, and optimal sensor health.
- **Threat Amber (`#F59E0B`)**: Unknown intercepts, frequency hopping anomalies, electronic countermeasure alerts, and non-critical hardware throttles.
- **Alarm Ruby (`#EF4444`)**: Directed energy alerts, active radar locks, electronic attack interference, and critical operational override states.
- **Text & Readout Rules**: Critical numeric readouts default to `#F1F5F9`. Units of measure, secondary coordinates, and non-actionable timestamps strictly use `#94A3B8` or `#475569` to preserve visual bandwidth.

## Typography

The dual-type hierarchy isolates human interaction from raw machine intelligence:
- **Command & Structural UI (Inter)**: Delivers rapid, friction-free optical parsing for operational navigation, panel headings, commands, and operator status. All section headers are set in full uppercase with expanded letter-spacing to reinforce structure.
- **Signal & Telemetry Stream (JetBrains Mono)**: Handles all scalar data, coordinates, decibel milliwatt (dBm) levels, frequency spectra (MHz/GHz), azimuth readouts, and hexadecimal payloads. Tabular numbers (`tnum`) and slashed zeros are strictly enforced to prevent vertical layout jitter during high-rate waterfall updates.

## Layout & Spacing

This design system uses a high-density, **Strict Mosaic Docking Grid** optimized for multi-monitor command workstations (1080p, 1440p, and 4K ultrawide arrays):
- **Base Grid**: Built upon an atomic 4px base coordinate system (`0.25rem`), enabling compact, pixel-locked packing of real-time scopes, azimuth wheels, and waterfall traces without wasted negative space.
- **Docking Structure**: Layouts conform to persistent top-line telemetry, customizable modular instrumentation docks, and an omnipresent tactical spectrum view. Docks separate via thin 1px radar-grid structural dividers with a rigid `dock-gap` of `0.25rem` (4px).
- **Reflow & Responsive Adaptations**:
  - *Console Primary (4K/Multi-display)*: 24-column dynamic grid. RF waterfall views span 12–16 columns, flanked by target tracking lists and hardware DSP status matrices.
  - *Field Terminal / Desktop (1920×1080)*: 12-column grid. Scopes collapse to standard aspect ratios (16:9, 4:3), with payload lists docking underneath rather than adjacent.
  - *Tactical Tablet (Field Deployments)*: 6-column single-stack layout. Complex RF waterfalls swap to condensed spectral envelope outlines with prioritized alarm cards.

## Elevation & Depth

This design system rejects conventional blurry drop shadows and floating skeuomorphic layers. Depth is structural, surgical, and low-profile:
- **Tonal Stepping**: Surface tiering indicates operational hierarchy. The raw canvas sits at Tier 0 (`#07080B`). Scopes and surveillance modules dock at Tier 1 (`#0B0D11`). Embedded widgets, channel selectors, and inspector drawers occupy Tier 2 (`#12161F`).
- **Precision Boundaries**: Rather than ambient diffusion, panels are bound by razor-sharp 1px structural borders (`#1E293B`). When a panel enters an active or targeted state, its boundary transitions to `#0E1E94` with a localized 2px micro-glow (`0 0 8px rgba(14, 30, 148, 0.25)`).
- **Tactical Overlays & Scanlines**: Secondary analytical instruments utilize an ultra-fine CRT raster overlay (`linear-gradient(rgba(18, 22, 31, 0) 50%, rgba(0, 0, 0, 0.35) 50%)`) with 2px line height, providing depth separation between hardware UI chrome and active RF sensor data.
- **Modal & Override Interceptors**: Critical command confirmation modals use Tier 3 (`#181E2A`) surfaces bounded by a 1px Threat Amber or Alarm Ruby border, backed by an 80% opacity `#07080B` backdrop veil with zero blur.

## Shapes

The design system enforces a **Zero-Radius Technical Doctrine (Sharp)**. 
- All standard containers, data cells, status tags, waterfall docks, and action switches feature absolute `0px` border-radii. Sharp 90-degree corners maximize addressable data real estate and align seamlessly against tactical coordinate markers.
- **Corner Notches / Chamfers**: Critical control hubs, weapon/countermeasure locks, and high-priority interception cards utilize a 4px diagonal clipped corner (chamfer) on the top-right or bottom-left edge to visually distinguish actionable directives from baseline passive readouts.
- **Crosshair Reticles & Technical Brackets**: Interactive viewports and lock pins use bracketed corner lines (`+` corner stamps, `L`-shaped crosshairs) constructed directly from 1px vector lines in `#334155` or `#0E1E94`.

## Components

### Buttons & Operational Commands
- **Primary Fire/Execute**: Boxy, uppercase, mono-labeled. High-contrast solid fill in `#0E1E94` with `#F1F5F9` text. On hover, background shifts to `#1e8dd5` with `#0B0D11` text.
- **Secondary / Engage**: Outlined with 1px `#334155` border, `#12161F` surface, and `#F1F5F9` text. Hover brings a 1px `#0E1E94` border and localized navy corner accents.
- **Destructive / Emergency Countermeasure**: Outlined in 1px `#EF4444`, background in 10% `#EF4444` tint. Text in `#EF4444`. Active state flashes high-contrast solid ruby with white mono text.

### Badges & RF Band Chips
- **RF Band Markers (e.g., VHF, UHF, SHF, X-BAND)**: Ultra-compact, uppercase `telemetry-sm` typography inside a 1px `#1E293B` frame. Background `#0B0D11`.
- **Classification / Threat Levels**:
  - `FRIENDLY / CLEARED`: Solid `#00BB7D` text, left-aligned 2px `#00BB7D` vertical bar indicator.
  - `SUSPECT`: Solid `#F59E0B` text, 2px amber bar, pulsing dot icon.
  - `HOSTILE / JAMMING`: Solid `#EF4444` inverted tag with black text for instant ocular capture.

### Lists & High-Density Telemetry Tables
- Built for continuous streaming data. Row heights are locked to a dense 24px.
- Alternate zebra stripping using `#0B0D11` and `#0E1118`.
- Column dividers are subtle 1px dashed or solid lines in `#1E293B`.
- Numeric data columns right-align using `JetBrains Mono` tabular figures. Instant visual delta indicators (e.g., `+0.4 dBm`, `-12 kHz`) render in semantic emerald or ruby.

### Form Inputs, Frequency Step Counters, & Toggles
- **Inputs**: Recessed `#07080B` surface, 1px `#1E293B` border, bright `#F1F5F9` mono text. Active focus transitions the border to `#0E1E94` with bracketed corners.
- **Frequency Adjusters / Spinners**: Segmented button groups displaying center frequency, bandwidth, and sweep speed, paired with fine/coarse incremental step buttons.
- **Checkboxes & Radios**: Angular square selectors (`0px` radius). Active state features a solid navy square centered within a 1px border frame.

### Instrumentation Cards & Spectral Scopes
- Enclosed in Tier 1 (`#0B0D11`) backdrops with 1px `#1E293B` boundaries.
- **Header Strip**: 20px tall header housing the module title (uppercase Inter, 9px, `#94A3B8`), status beacon (e.g., `LIVE SWEEP 120 MS`), and coordinate crosshair stamps.
- **Interactive Scopes**: Waterfall visualizers and FFT fast Fourier transform plots contain calibrated decibel scale overlays (`telemetry-sm`) set at `#475569`, tracking reticles, and lock vectors rendered in primary navy or ruby.