/**
 * useOverviewTelemetry
 *
 * Single source-of-truth hook for the Mission Overview page.
 * Combines WebSocket /ws/state (4 Hz) + REST poll /telemetry/latest
 * and /mission/status (2 Hz fallback).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { backend } from "./backend";

const TOTAL_BANDS = 36;
const BAND_WIDTH_MHZ = 500;
const IBW_MHZ = 1000;
const DWELL_HISTORY_SIZE = 6;

const MODE_LABELS = {
  0: "SHORT_DWELL",
  1: "NORMAL_DWELL",
  2: "LONG_DWELL",
  3: "REVISIT",
  4: "PREEMPTIVE_INTERCEPT",
};

function bandToFreqMHz(band) {
  return band * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2;
}

function buildDefault() {
  return {
    wsStatus: "OFFLINE",
    live: false,
    source: "none",
    totalSpectrumGHz: 18,
    ibwMHz: IBW_MHZ,
    activeBands: 0,
    quietBands: TOTAL_BANDS,
    currentBand: 16,
    currentFreqMHz: 8250,
    currentMode: "—",
    currentDwellUs: 0,
    totalHits: 0,
    totalDwells: 0,
    rollingPd: 0,
    rollingMedianLatencyUs: 0,
    missionClockUs: 0,
    missionActive: false,
    bandHeights: Array(TOTAL_BANDS).fill(0).map((_, i) => (i === 16 ? 0.94 : 0.03)),
    bandStates: Array(TOTAL_BANDS).fill("quiet").map((_, i) => (i === 16 ? "active" : "quiet")),
    scheduler: {
      chosenBand: 16,
      chosenFreqMHz: 8250,
      scanMode: "—",
      dwellUs: 0,
      drqnScore: 0,
      interceptProbability: 0,
      predictedEtaUs: 0,
      actionSpace: 180,
      decisionReason: "—",
      explorationPressure: 0,
      qMargin: 0,
      moeGating: 0,
    },
    dwellHistory: [],
    cognitiveExplanation: {},
    systemMetrics: {},
    pdws: [],
  };
}

function processTelemetry(raw, missionStat) {
  const next = buildDefault();

  if (missionStat) {
    next.missionActive = missionStat.is_mission_active ?? false;
    next.missionClockUs = missionStat.mission_clock_us ?? 0;
    next.totalDwells = missionStat.total_dwells ?? 0;
    next.totalHits = missionStat.total_hits ?? 0;
    next.rollingPd = missionStat.rolling_pd ?? 0;
    next.rollingMedianLatencyUs = missionStat.rolling_median_latency_us ?? 0;
  }

  if (!raw || raw.live === false) {
    next.live = false;
    next.source = raw?.source ?? "none";
    return next;
  }

  next.live = true;
  next.source = raw.source ?? "publisher";

  const m = raw.metrics ?? raw;

  const band = m.band ?? 0;
  const modeIdx = m.mode ?? 1;
  const modeName = m.mode_name ?? MODE_LABELS[modeIdx] ?? "NORMAL_DWELL";
  const dwellUs = m.dwell_time_us ?? 0;
  const clockUs = m.clock_us ?? 0;

  next.currentBand = band;
  next.currentFreqMHz = bandToFreqMHz(band);
  next.currentMode = modeName;
  next.currentDwellUs = dwellUs;
  if (!missionStat) next.missionClockUs = clockUs;

  const ce = m.cognitive_explanation ?? {};
  next.cognitiveExplanation = ce;

  const sm = m.system_metrics ?? {};
  next.systemMetrics = sm;
  if (!missionStat) {
    next.rollingPd = sm.rolling_pd ?? 0;
    next.rollingMedianLatencyUs = sm.rolling_median_latency_us ?? 0;
  }

  const drqnScore = ce.drqn_score ?? 0;
  const interceptProb = ce.prediction_confidence ?? 0;
  const etaUs = ce.predicted_eta_us ?? 0;
  const reason = ce.decision_reason ?? "—";
  const explorationPressure = ce.exploration_pressure ?? 0;
  const qMargin = ce.q_margin ?? 0;

  next.scheduler = {
    chosenBand: band,
    chosenFreqMHz: next.currentFreqMHz,
    scanMode: modeName,
    dwellUs: dwellUs,
    drqnScore: drqnScore,
    interceptProbability: interceptProb,
    predictedEtaUs: Math.max(0, etaUs),
    actionSpace: 180,
    decisionReason: reason,
    explorationPressure: explorationPressure,
    qMargin: qMargin,
    moeGating: 1 - explorationPressure,
  };

  // Each time show only the band the receiver is tuned into
  next.bandStates = Array(TOTAL_BANDS).fill("quiet").map((_, i) =>
    i === band ? "active" : "quiet"
  );
  next.bandHeights = Array(TOTAL_BANDS).fill(0).map((_, i) =>
    i === band ? 0.94 : 0.03
  );

  const activeCount = 1;
  next.activeBands = activeCount;
  next.quietBands = TOTAL_BANDS - activeCount;

  const rawPdws = raw.pdws ?? m.pdws ?? [];
  next.pdws = Array.isArray(rawPdws) ? rawPdws : [];

  return next;
}

export function useOverviewTelemetry() {
  const [state, setState] = useState(buildDefault);
  const dwellHistoryRef = useRef([]);
  const missionStatRef = useRef(null);
  const latestTelRef = useRef(null);

  const ingest = useCallback((rawTelemetry, missionStat) => {
    if (missionStat) missionStatRef.current = missionStat;
    if (rawTelemetry) latestTelRef.current = rawTelemetry;

    const resolved = processTelemetry(
      latestTelRef.current,
      missionStatRef.current,
    );

    if (resolved.live) {
      const m = (latestTelRef.current?.metrics ?? latestTelRef.current) ?? {};
      const band = m.band ?? resolved.currentBand;
      const hit = m.hit ?? false;
      const dwellUs = m.dwell_time_us ?? resolved.currentDwellUs;
      const clockUs = m.clock_us ?? resolved.missionClockUs;
      const modeName = m.mode_name ?? resolved.currentMode;

      const entry = {
        id: `${clockUs}-${band}`,
        band,
        freqMHz: bandToFreqMHz(band),
        mode: modeName,
        hit,
        dwellUs,
        clockUs,
        now: true,
      };

      const prev = dwellHistoryRef.current;
      const isDuplicate = prev.length > 0 && prev[prev.length - 1].id === entry.id;
      if (!isDuplicate) {
        const updated = prev.map((e) => ({ ...e, now: false }));
        updated.push(entry);
        dwellHistoryRef.current = updated.slice(-DWELL_HISTORY_SIZE);
      }
    }

    resolved.dwellHistory = dwellHistoryRef.current;
    setState(resolved);
  }, []);

  // WebSocket – primary 4 Hz real-time feed
  useEffect(() => {
    let unmounted = false;

    const ws = backend.connectState({
      onOpen: () => {
        if (!unmounted) setState((p) => ({ ...p, wsStatus: "ONLINE" }));
      },
      onMessage: (payload) => {
        if (unmounted) return;
        setState((p) => ({ ...p, wsStatus: "ONLINE" }));
        ingest(payload, missionStatRef.current);
      },
      onError: () => {
        if (!unmounted) setState((p) => ({ ...p, wsStatus: "ERROR" }));
      },
      onClose: () => {
        if (!unmounted) setState((p) => ({ ...p, wsStatus: "OFFLINE" }));
      },
    });

    return () => {
      unmounted = true;
      ws.close();
    };
  }, [ingest]);

  // REST poll fallback – 2 Hz
  useEffect(() => {
    let active = true;

    async function poll() {
      try {
        const [telRes, statRes] = await Promise.allSettled([
          backend.api.getLatestTelemetry(),
          backend.api.getMissionStatus(),
        ]);
        if (!active) return;
        const tel = telRes.status === "fulfilled" ? telRes.value : null;
        const stat = statRes.status === "fulfilled" ? statRes.value : null;
        ingest(tel, stat);
      } catch {
        // silently swallow
      }
    }

    poll();
    const id = setInterval(poll, 500);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [ingest]);

  return state;
}
