export const mockSystem = {
  spectrum: {
    totalBandwidthMHz: 18000,
    instantaneousBandwidthMHz: 1000,
    bandCount: 36,
    bandWidthMHz: 500,
    currentTuneMHz: 8250,
    currentBand: 16,
  },

  receiver: {
    status: "READY",
    detectionThresholdDb: 15,
    frequencyStepMHz: 500,
    currentMode: "REVISIT",
    dwellTimeUs: 120,
    detections: 3412,
  },

  scheduler: {
    status: "READY",
    architecture: "DRQN + MoE",
    observationDimension: 360,
    actionCount: 180,
    selectedBand: 16,
    selectedMode: "REVISIT",
    selectedFrequencyMHz: 8250,
    score: 0.941,
    predictedInterceptProbability: 0.88,
    predictedInterceptTimeUs: 42,
  },

  mission: {
    activeBands: 14,
    quietBands: 22,
    interceptions: 842,
    falseAlarms: 23,
    hits: 819,
    misses: 96,
  },

  health: {
    rfEnvironment: "ONLINE",
    receiver: "ONLINE",
    pdwDetector: "ONLINE",
    observationPipeline: "ONLINE",
    scheduler: "READY",
    dataset: "VALID",
  },

  timeline: [
    {
      band: 6,
      frequencyMHz: 3250,
      mode: "NORMAL_DWELL",
      offsetMs: -500,
      result: "MISS",
    },
    {
      band: 10,
      frequencyMHz: 5250,
      mode: "SHORT_DWELL",
      offsetMs: -400,
      result: "SEARCH",
    },
    {
      band: 16,
      frequencyMHz: 8250,
      mode: "REVISIT",
      offsetMs: -120,
      result: "HIT",
    },
    {
      band: 28,
      frequencyMHz: 14250,
      mode: "LONG_DWELL",
      offsetMs: -40,
      result: "SEARCH",
    },
    {
      band: 16,
      frequencyMHz: 8250,
      mode: "PREEMPTIVE_INTERCEPT",
      offsetMs: 0,
      result: "ARMED",
    },
  ],
};

export const syntheticSystem = mockSystem;