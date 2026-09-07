import React, { useEffect, useState } from "react";
import { syntheticSystem } from "../data/mockSystem";
import { loadMissionOverview } from "../services/missionService";
import SpectrumView from "../components/Spectrum/SpectrumView";

function MetricCard({ label, value, unit, status }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>

      <div className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </div>

      {status && (
        <div className="metric-status">
          {status}
        </div>
      )}
    </div>
  );
}

function SpectrumMap() {
  const {
    totalBandwidthMHz,
    currentTuneMHz,
    instantaneousBandwidthMHz,
    currentBand,
  } = syntheticSystem.spectrum;

  return (
    <div className="spectrum-panel panel">
      <div className="panel-header">
        <div>
          <div className="panel-kicker">
            RF ENVIRONMENT
          </div>

          <h2>0–18 GHz Wideband Spectrum</h2>
        </div>

        <div className="panel-badge">
          1 GHz IBW LOCKED
        </div>
      </div>

      <SpectrumView
        currentTuneMHz={currentTuneMHz}
        totalBandwidthMHz={totalBandwidthMHz}
        instantaneousBandwidthMHz={
          instantaneousBandwidthMHz
        }
        selectedBand={currentBand}
      />

      <div className="spectrum-footer">
        <div>
          <span>CURRENT TUNE</span>

          <strong>
            {currentTuneMHz.toLocaleString()} MHz
          </strong>
        </div>

        <div>
          <span>IBW</span>

          <strong>
            {instantaneousBandwidthMHz} MHz
          </strong>
        </div>

        <div>
          <span>BAND</span>

          <strong>B{currentBand}</strong>
        </div>
      </div>
    </div>
  );
}

function SchedulerDecision() {
  const scheduler = syntheticSystem.scheduler;

  return (
    <div className="decision-panel panel">
      <div className="panel-header">
        <div>
          <div className="panel-kicker">
            SMART SCHEDULER
          </div>

          <h2>Current Decision</h2>
        </div>

        <div className="live-badge">
          DRQN ACTIVE
        </div>
      </div>

      <div className="decision-main">
        <div className="decision-frequency">
          <span>CHOSEN BAND</span>

          <strong>
            B{scheduler.selectedBand}
          </strong>

          <small>
            {scheduler.selectedFrequencyMHz.toLocaleString()} MHz
          </small>
        </div>

        <div className="decision-mode">
          <span>SCAN MODE</span>

          <strong>
            {scheduler.selectedMode}
          </strong>

          <small>
            {scheduler.dwellTimeUs} µs dwell
          </small>
        </div>
      </div>

      <div className="decision-grid">
        <div>
          <span>ACTION SCORE</span>

          <strong>
            {scheduler.score.toFixed(3)}
          </strong>
        </div>

        <div>
          <span>INTERCEPT PROBABILITY</span>

          <strong>
            {(
              scheduler.predictedInterceptProbability * 100
            ).toFixed(1)}
            %
          </strong>
        </div>

        <div>
          <span>PREDICTED TIME</span>

          <strong>
            {scheduler.predictedInterceptTimeUs} µs
          </strong>
        </div>

        <div>
          <span>ACTION SPACE</span>

          <strong>
            {scheduler.actionCount}
          </strong>
        </div>
      </div>

      <div className="decision-reasoning">
        <div className="reasoning-title">
          DECISION REASONING
        </div>

        <div className="reasoning-row">
          <span>Primary Driver</span>
          <strong>Revisit pressure</strong>
        </div>

        <div className="reasoning-row">
          <span>Secondary Driver</span>
          <strong>Recent pulse activity</strong>
        </div>

        <div className="reasoning-row">
          <span>Observation</span>
          <strong>360-D state</strong>
        </div>
      </div>
    </div>
  );
}

function Timeline() {
  return (
    <div className="timeline-panel panel">
      <div className="panel-header">
        <div>
          <div className="panel-kicker">
            TEMPORAL STATE
          </div>

          <h2>Dwell & Intercept Timeline</h2>
        </div>
      </div>

      <div className="timeline">
        {syntheticSystem.timeline.map(
          (event, index) => (
            <div
              className="timeline-event"
              key={`${event.band}-${event.frequencyMHz}-${index}`}
            >
              <div className="timeline-time">
                {event.offsetMs === 0
                  ? "NOW"
                  : `T ${event.offsetMs} ms`}
              </div>

              <div className="timeline-line">
                <div className="timeline-dot" />
              </div>

              <div className="timeline-content">
                <strong>
                  B{event.band} ·{" "}
                  {event.frequencyMHz.toLocaleString()} MHz
                </strong>

                <span>
                  {event.mode}
                </span>

                <small>
                  {event.result}
                </small>
              </div>
            </div>
          ),
        )}
      </div>
    </div>
  );
}

export default function MissionOverview() {
  const {
    spectrum,
    receiver,
    mission,
  } = syntheticSystem;

  const [backendData, setBackendData] =
    useState(null);

  useEffect(() => {
    let active = true;

    async function loadData() {
      try {
        const data =
          await loadMissionOverview();

        if (!active) {
          return;
        }

        setBackendData(data);
      } catch {
        if (!active) {
          return;
        }

        setBackendData(null);
      }
    }

    loadData();

    return () => {
      active = false;
    };
  }, []);

  const usingBackend =
    backendData?.connected === true;

  return (
    <div className="mission-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            MISSION CONTROL
          </div>

          <h1>
            Smart Scan Mission Overview
          </h1>

          <p>
            Intelligent frequency and dwell
            selection across a wideband RF
            environment.
          </p>
        </div>

        <div className="mission-state">
          <span className="status-dot" />
          SYSTEM ACTIVE
        </div>
      </div>

      <div className="mission-data-status">
        <span
          className={`connection-dot ${
            usingBackend
              ? "connected"
              : "synthetic"
          }`}
        />

        <span>
          {usingBackend
            ? "BACKEND DATA"
            : "SYNTHETIC RF DATA"}
        </span>
      </div>

      <div className="metric-grid">
        <MetricCard
          label="TOTAL SPECTRUM"
          value="18.00"
          unit="GHz"
          status="0–18,000 MHz"
        />

        <MetricCard
          label="INSTANTANEOUS BANDWIDTH"
          value="1.00"
          unit="GHz"
          status="Receiver IBW"
        />

        <MetricCard
          label="ACTIVE BANDS"
          value={
            spectrum.bandCount -
            mission.quietBands
          }
          unit="/ 36"
          status="Currently active"
        />

        <MetricCard
          label="CURRENT TUNE"
          value={spectrum.currentTuneMHz.toLocaleString()}
          unit="MHz"
          status={`Band ${spectrum.currentBand}`}
        />

        <MetricCard
          label="PDW DETECTIONS"
          value={receiver.detections.toLocaleString()}
          status="Current mission"
        />

        <MetricCard
          label="INTERCEPTIONS"
          value={mission.interceptions.toLocaleString()}
          status={`${mission.hits.toLocaleString()} hits`}
        />

        <MetricCard
          label="CURRENT MODE"
          value={receiver.currentMode}
          status={`${receiver.dwellTimeUs} µs dwell`}
        />
      </div>

      <div className="overview-grid">
        <SpectrumMap />
        <SchedulerDecision />
      </div>

      <Timeline />
    </div>
  );
}