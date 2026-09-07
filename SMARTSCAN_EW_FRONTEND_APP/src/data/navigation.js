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
      { id: "overview", label: "Mission Overview" },
      { id: "spectrum", label: "Live Spectrum" },
      { id: "smart-scan", label: "Smart Scan" },
    ],
  },
  {
    section: "RECEIVER",
    items: [
      { id: "receiver", label: "Receiver / PDW" },
      { id: "emitters", label: "Emitters / Truth" },
      { id: "interception", label: "Interception" },
    ],
  },
  {
    section: "TRAINING & DATA",
    items: [
      { id: "training", label: "Training" },
      { id: "data", label: "Data & Explorer" },
      { id: "experiments", label: "Experiments" },
      { id: "performance", label: "Performance" },
      { id: "replay", label: "Replay" },
    ],
  },
  {
    section: "SYSTEM",
    items: [
      { id: "system", label: "System / Config" },
    ],
  },
];