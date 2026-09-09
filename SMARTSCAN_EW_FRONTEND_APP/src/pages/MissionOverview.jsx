import { useEffect, useState } from "react";
import { syntheticSystem } from "../data/mockSystem";
import { loadMissionOverview } from "../services/missionService";
import { api } from "../services/api";
import {
  BandMatrix,
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
  const [missionStatus, setMissionStatus] = useState(null);

  useEffect(() => {
    let active = true;
    async function loadData() {
      try {
        const [data, stat] = await Promise.allSettled([
          loadMissionOverview(),
          api.getMissionStatus(),
        ]);
        if (!active) return;
        if (data.status === "fulfilled") setBackendData(data.value);
        if (stat.status === "fulfilled") setMissionStatus(stat.value);
      } catch {
        if (!active) return;
      }
    }
    loadData();
    const interval = setInterval(loadData, 1000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const usingBackend = backendData?.connected === true || (missionStatus && missionStatus.total_dwells > 0);
  const activeBands = spectrum.bandCount - mission.quietBands;

  const hasLive = missionStatus && missionStatus.total_dwells > 0;
  const liveHits = hasLive ? missionStatus.total_hits : receiver.detections;
  const liveInterceptions = hasLive ? `${(missionStatus.rolling_pd * 100).toFixed(1)}% Pd` : mission.interceptions.toLocaleString();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="grid_view" title="SMART SCAN MISSION OVERVIEW" badge="SUM" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Intelligent frequency and dwell selection across a wideband RF
          environment. <DataSourceBadge connected={usingBackend} />
          {hasLive && <span style={{ color: "#49df9d", marginLeft: 8 }}>● Live Operational Mission ({missionStatus.total_dwells} Dwells)</span>}
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Mission KPI strip">
        <KpiCard label="TOTAL SPECTRUM" icon="tune" value="18.00" unit="GHz" footLeft="0.00 MHz" footRight="18,000 MHz" valueColor="#bdc2ff" />
        <KpiCard label="INSTANTANEOUS BW" icon="cell_tower" value="500" unit="MHz" footLeft="CANONICAL IBW" footRight="TUNABLE" valueColor="#96ccff" />
        <KpiCard label="ACTIVE BANDS" icon="sensors" value={`${activeBands}`} unit="/ 36" footLeft="38.8% OCCUPIED" footRight="22 QUIET" valueColor="#49df9d" />
        <KpiCard label="TOTAL HITS" icon="grain" value={liveHits.toLocaleString()} unit="PULSES" footLeft={hasLive ? `DWELLS: ${missionStatus.total_dwells}` : "+128/s"} footRight="CONFIRMED" valueColor="#e2e2e8" />
        <KpiCard label="INTERCEPT RATE" icon="verified" value={liveInterceptions} unit="RATE" footLeft={hasLive ? `LAT: ${missionStatus.rolling_median_latency_us.toFixed(0)}µs` : "TRACK LOCK"} footRight="0 FALSE POS" valueColor="#6afcb8" />
        <KpiCard label="CURRENT MODE" icon="neurology" value={hasLive ? (backendData?.telemetry?.mode_name ?? "NORMAL_DWELL") : receiver.currentMode} unit="500µs" footLeft="SCHED: DRQN" footRight="PRIO: TIER-1" valueColor="#96ccff" />
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
            <PanelHead icon="psychology" title="CURRENT SCHEDULER DECISION" badge="DRQN + MoE ACTIVE" badgeColor="#bdc2ff" />
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
          </div>
        </div>
      </div>
    </div>
  );
}
