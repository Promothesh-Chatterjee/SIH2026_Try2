const SYNTHETIC_BANDS = [
  6,
  10,
  16,
  28,
];

const SCAN_MODES = [
  "NORMAL_DWELL",
  "SHORT_DWELL",
  "REVISIT",
  "LONG_DWELL",
  "PREEMPTIVE_INTERCEPT",
];

const EVENT_TYPES = [
  "SEARCH",
  "DETECTION",
  "HIT",
  "ARMED",
];

function randomChoice(values) {
  return values[
    Math.floor(
      Math.random() *
        values.length,
    )
  ];
}

function randomFloat(
  min,
  max,
  decimals = 2,
) {
  const value =
    min +
    Math.random() *
      (max - min);

  return Number(
    value.toFixed(decimals),
  );
}

export function createSyntheticTelemetry(
  currentBand = 16,
) {
  const band =
    SYNTHETIC_BANDS.includes(
      currentBand,
    )
      ? currentBand
      : randomChoice(
          SYNTHETIC_BANDS,
        );

  const frequencyMHz =
    band * 500 + 250;

  const eventType =
    randomChoice(
      EVENT_TYPES,
    );

  const mode =
    randomChoice(
      SCAN_MODES,
    );

  const detected =
    eventType === "DETECTION" ||
    eventType === "HIT";

  const intercepted =
    eventType === "HIT";

  const dwellTimeUs =
    mode === "SHORT_DWELL"
      ? 50
      : mode === "LONG_DWELL"
        ? 250
        : 100;

  return {
    timestamp: Date.now(),

    source: "synthetic",

    valid: true,

    live: false,

    message:
      "Synthetic RF telemetry",

    receiver: {
      centerFrequencyMHz:
        frequencyMHz,

      ibwMHz: 1000,

      dwellTimeUs,

      thresholdDb: 15,

      band,
    },

    scheduler: {
      selectedBand: band,

      selectedFrequencyMHz:
        frequencyMHz,

      mode,

      score: randomFloat(
        0.55,
        0.99,
        3,
      ),

      predictedInterceptProbability:
        randomFloat(
          0.35,
          0.95,
          3,
        ),

      predictedInterceptTimeUs:
        randomFloat(
          100,
          600,
          1,
        ),
    },

    events: [
      {
        timeUs:
          randomFloat(
            0,
            1000,
            1,
          ),

        band,

        frequencyMHz:
          randomFloat(
            frequencyMHz - 8,
            frequencyMHz + 8,
            2,
          ),

        type: eventType,

        mode,

        result:
          intercepted
            ? "INTERCEPT"
            : detected
              ? "DETECTED"
              : "NO_DETECTION",
      },
    ],

    detections:
      detected ? 1 : 0,

    interceptions:
      intercepted ? 1 : 0,
  };
}