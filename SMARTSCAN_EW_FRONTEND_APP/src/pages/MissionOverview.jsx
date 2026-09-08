import { useEffect, useState } from "react";
import { syntheticSystem } from "../data/mockSystem";
import { loadMissionOverview } from "../services/missionService";
import {
  BandMatrix,
  CandidateActions,
  CmdBadge,
  DataSourceBadge,
  DwellTimeline,
  KpiCard,
  PanelHead,
  PipelineFlow,
} from "../components/stitch";

export default function MissionOverview() {
  const { spectrum, receiver, mission, scheduler } = syntheticSystem;
  const [backendData, setBackendData] = useState(null);

  useEffect(() => {
    let active = true;
    async function loadData() {
      try {
        const data = await loadMissionOverview();
        if (!active) return;
        setBackendData(data);
      } catch {
        if (!active) return;
        setBackendData(null);
      }
    }
    loadData();
    return () => {
      active = false;
    };
  }, []);

  const usingBackend = backendData?.connected === true;
  const activeBands = spectrum.bandCount - mission.quietBands;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="grid_view" title="SMART SCAN MISSION OVERVIEW" badge="SUM" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Intelligent frequency and dwell selection across a wideband RF
          environment. <DataSourceBadge connected={usingBackend} />
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Mission KPI strip">
        <KpiCard label="TOTAL SPECTRUM" icon="tune" value="18.00" unit="GHz" footLeft="0.00 MHz" footRight="18,000 MHz" valueColor="#bdc2ff" />
        <KpiCard label="INSTANTANEOUS BW" icon="cell_tower" value="1.00" unit="GHz" footLeft="STEP Δ: 500 MHz" footRight="TUNABLE" valueColor="#96ccff" />
        <KpiCard label="ACTIVE BANDS" icon="sensors" value={`${activeBands}`} unit="/ 36" footLeft="38.8% OCCUPIED" footRight="22 QUIET" valueColor="#49df9d" />
        <KpiCard label="CURRENT TUNE" icon="file_download_done" value={spectrum.currentTuneMHz.toLocaleString()} unit="MHz" footLeft={`BAND ${spectrum.currentBand}`} footRight="8000–8500 MHz" valueColor="#bdc2ff" />
        <KpiCard label="DETECTIONS" icon="grain" value={receiver.detections.toLocaleString()} unit="PDW" footLeft="+128/s" footRight="VALID SNR ≥15dB" valueColor="#e2e2e8" />
        <KpiCard label="INTERCEPTIONS" icon="verified" value={mission.interceptions.toLocaleString()} unit="EVTS" footLeft="TRACK LOCK 98.2%" footRight="0 FALSE POS" valueColor="#6afcb8" />
        <KpiCard label="CURRENT MODE" icon="neurology" value={receiver.currentMode} unit={`${receiver.dwellTimeUs}µs`} footLeft="SCHED: DRQN" footRight="PRIO: TIER-1" valueColor="#96ccff" />
      </section>

      <PipelineFlow />

      <div className="st-grid-12">
        <div className="st-span-8" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="show_chart"
              title="0–18 GHz WIDEBAND RF EMISSION & IBW SCANNING MAP"
              badge="CALIBRATED FFT"
              badgeColor="#96ccff"
            />
            <div className="st-tsm" style={{ display: "flex", gap: 8, color: "#908f9e" }}>
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#96ccff" }} /> STABLE EMITTER</span>
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#49df9d" }} /> INTERCEPTED</span>
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#3097e0" }} /> AGILE TRACK</span>
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#333539" }} /> QUIET</span>
            </div>
            <BandMatrix tuneBand={spectrum.currentBand} />
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#908f9e" }}>CURRENT TUNE</span>
              <strong style={{ color: "#e2e2e8" }}>
                {spectrum.currentTuneMHz.toLocaleString()} MHz · B{spectrum.currentBand} · IBW {spectrum.instantaneousBandwidthMHz} MHz
              </strong>
            </div>
          </div>
          <DwellTimeline />
        </div>

        <div className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead icon="psychology" title="CURRENT SCHEDULER DECISION" badge="DRQN ACTIVE" badgeColor="#bdc2ff" />
            <div style={{ background: "#1a1c20", border: "1px solid #454653", padding: 6, display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>CHOSEN TARGET</span>
                <strong className="st-tmd" style={{ color: "#bdc2ff" }}>
                  BAND {scheduler.selectedBand} ({scheduler.selectedFrequencyMHz.toLocaleString()} MHz)
                </strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>SCAN MODE</span>
                <CmdBadge color="#96ccff">{scheduler.selectedMode} ({receiver.dwellTimeUs} µs)</CmdBadge>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>ACTION SCORE</span>
                <strong className="st-tmd">{scheduler.score.toFixed(3)}</strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>INTERCEPT PROBABILITY</span>
                <strong className="st-tmd">{(scheduler.predictedInterceptProbability * 100).toFixed(1)}%</strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>PREDICTED TIME</span>
                <strong className="st-tmd">{scheduler.predictedInterceptTimeUs} µs</strong>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>ACTION SPACE</span>
                <strong className="st-tmd">{scheduler.actionCount}</strong>
              </div>
            </div>

            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
              <span className="st-headline">REASONING & UTILITY DECOMPOSITION</span>
              <span>MoE GATING 0.992</span>
            </div>
            <div style={{ background: "#1a1c20", border: "1px solid #454653", padding: 6, display: "flex", flexDirection: "column", gap: 6 }}>
              <div className="st-body" style={{ color: "#c6c5d5" }}>
                <strong style={{ color: "#e2e2e8" }}>Primary Driver: Revisit Timeout. </strong>
                Threshold exceeded on radar threat track <strong style={{ color: "#bdc2ff" }}>TRK-084</strong> (+20 ms over limit).
              </div>
              <div className="st-body" style={{ color: "#c6c5d5" }}>
                <strong style={{ color: "#e2e2e8" }}>Secondary Driver: Pulse Train Agility. </strong>
                Agility 0.78 with predicted imminent PRI emission in <strong style={{ color: "#96ccff" }}>[T+45 µs]</strong>.
              </div>
              <div className="st-body" style={{ color: "#c6c5d5" }}>
                <strong style={{ color: "#e2e2e8" }}>Uncertainty Delta. </strong>
                Entropy +14% over previous 3 dwell cycles. Immediate intercept mandated.
              </div>
            </div>

            <div className="st-tsm" style={{ background: "#1a1c20", border: "1px solid #454653", padding: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "#908f9e" }}>DRQN LSTM MEMORY</span>
                <span style={{ color: "#49df9d" }}>WARM STATE [L-HIDDEN 256]</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <span>EXPERT 2: PERIODIC TRACKS</span>
                <strong style={{ color: "#bdc2ff" }}>W: 0.64</strong>
              </div>
              <div style={{ height: 6, background: "#333539" }}>
                <div style={{ width: "64%", height: "100%", background: "#bdc2ff" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <span>EXPERT 4: AGILE INTERCEPT</span>
                <strong style={{ color: "#96ccff" }}>W: 0.36</strong>
              </div>
              <div style={{ height: 6, background: "#333539" }}>
                <div style={{ width: "36%", height: "100%", background: "#96ccff" }} />
              </div>
            </div>

            <CandidateActions />
          </div>
        </div>
      </div>
    </div>
  );
}
