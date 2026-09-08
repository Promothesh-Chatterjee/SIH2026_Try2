export const modes = [
  {
    id: "live",
    label: "LIVE / OPERATION",
  },
  {
    id: "training",
    label: "TRAINING",
  },
  {
    id: "replay",
    label: "REPLAY / ANALYSIS",
  },
];

export const navigation = [
  {
    section: "MISSION",
    items: [
      { id: "overview", label: "OVERVIEW", badge: "SUM" },
      { id: "spectrum", label: "LIVE SPECTRUM", badge: "18GHz" },
      { id: "smart-scan", label: "SMART SCAN", badge: "DRQN" },
    ],
  },
  {
    section: "RECEIVER",
    items: [
      { id: "receiver", label: "RECEIVER", badge: "PDW" },
      { id: "emitters", label: "EMITTERS", badge: "TRUTH" },
      { id: "interception", label: "INTERCEPTION", badge: "TxF" },
    ],
  },
  {
    section: "EVALUATION",
    items: [
      { id: "performance", label: "PERFORMANCE", badge: "EVAL" },
      { id: "dataset", label: "DATASET", badge: "AUDIT" },
    ],
  },
  {
    section: "EXTENSIONS",
    items: [
      { id: "data", label: "Data Explorer", badge: "EXT" },
      { id: "training", label: "Training", badge: "EXT" },
      { id: "rewards", label: "Rewards", badge: "EXT" },
      { id: "experiments", label: "Experiments", badge: "EXT" },
      { id: "replay", label: "Replay", badge: "EXT" },
    ],
  },
  {
    section: "SYSTEM",
    items: [
      { id: "system", label: "SYSTEM / CONFIG", badge: "SYS" },
    ],
  },
];