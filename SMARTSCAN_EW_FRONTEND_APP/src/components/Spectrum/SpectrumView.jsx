const DEFAULT_ACTIVE_BANDS = new Set([6, 10, 16, 28]);

export default function SpectrumView({
  currentTuneMHz = 8250,
  totalBandwidthMHz = 18000,
  instantaneousBandwidthMHz = 1000,
  selectedBand = 16,
  activeBands = DEFAULT_ACTIVE_BANDS,
  showLabels = true,
}) {
  const bandCount = 36;

  const safeTotal = Math.max(totalBandwidthMHz, 1);
  const safeIBW = Math.min(
    Math.max(instantaneousBandwidthMHz, 0),
    safeTotal
  );

  const windowStart = Math.max(
    0,
    Math.min(
      100,
      ((currentTuneMHz - safeIBW / 2) / safeTotal) * 100
    )
  );

  const windowWidth = (safeIBW / safeTotal) * 100;

  const bands = Array.from({ length: bandCount }, (_, index) => index);

  return (
    <div className="spectrum-view">
      <div className="spectrum-scale">
        <span>0 GHz</span>
        <span>4 GHz</span>
        <span>8 GHz</span>
        <span>12 GHz</span>
        <span>18 GHz</span>
      </div>

      <div className="spectrum-track">
        {bands.map((band) => {
          const isSelected = band === selectedBand;
          const isActive = activeBands.has
            ? activeBands.has(band)
            : activeBands.includes?.(band);

          const deterministicHeight =
            25 + ((band * 17) % 60);

          return (
            <div
              key={band}
              className={[
                "spectrum-band",
                isActive ? "active" : "",
                isSelected ? "selected" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              title={`Band ${band} · ${
                band * 500
              }–${(band + 1) * 500} MHz`}
            >
              <div
                className="spectrum-bar"
                style={{
                  height: `${deterministicHeight}%`,
                }}
              />

              {showLabels && <span>B{band}</span>}
            </div>
          );
        })}

        <div
          className="receiver-window"
          style={{
            left: `${windowStart}%`,
            width: `${windowWidth}%`,
          }}
        >
          <span>1 GHz RX WINDOW</span>
        </div>
      </div>
    </div>
  );
}