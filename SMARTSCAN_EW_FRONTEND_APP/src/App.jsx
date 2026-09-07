import AppLayout from "./layouts/AppLayout";
import MissionOverview from "./pages/MissionOverview";
import LiveSpectrum from "./pages/LiveSpectrum";
import SmartScan from "./pages/SmartScan";
import Receiver from "./pages/Receiver";
import Interception from "./pages/Interception";
import Emitters from "./pages/Emitters";
import DataExplorer from "./pages/DataExplorer";
import TrainingMonitor from "./pages/TrainingMonitor";
import RewardMonitor from "./pages/RewardMonitor";
import ExperimentHistory from "./pages/ExperimentHistory";
import ReplayAnalysis from "./pages/ReplayAnalysis";
import "./App.css";

function PlaceholderPage({ activePage, mode }) {
  const titles = {
    "smart-scan": "Smart Scan Decision Engine",
    receiver: "Receiver / PDW",
    emitters: "Emitters / Simulation Truth",
    interception: "Interception",
    training: "Training",
    data: "Data & Explorer",
    experiments: "Experiments",
    performance: "Performance",
    replay: "Replay / Analysis",
    system: "System / Configuration",
  };

  return (
    <section className="page-placeholder">
      <div className="page-eyebrow">{mode.toUpperCase()}</div>

      <h1>{titles[activePage] ?? "SmartScan EW"}</h1>

      <p>
        This module is connected to the unified SmartScan EW
        application shell and will be integrated next.
      </p>

      <div className="placeholder-grid">
        <div className="placeholder-card">
          <span>ACTIVE PAGE</span>
          <strong>{activePage}</strong>
        </div>

        <div className="placeholder-card">
          <span>SYSTEM MODE</span>
          <strong>{mode}</strong>
        </div>

        <div className="placeholder-card">
          <span>RF SPECTRUM</span>
          <strong>18 GHz</strong>
        </div>

        <div className="placeholder-card">
          <span>RECEIVER IBW</span>
          <strong>1 GHz</strong>
        </div>
      </div>
    </section>
  );
}

export default function App() {
  return (
    <AppLayout>
      {({ activePage, mode }) => {
        if (activePage === "overview") {
          return <MissionOverview />;
        }

        if (activePage === "spectrum") {
          return <LiveSpectrum />;
        }

        if (activePage === "smart-scan") {
          return <SmartScan />;
        }

        if (activePage === "receiver") {
          return <Receiver />;
        }

        if (activePage === "interception") {
          return <Interception />;
        }

        if (activePage === "emitters") {
          return <Emitters />;
        }

        if (activePage === "data") {
          return <DataExplorer />;
        }

        if (activePage === "training") return <TrainingMonitor />;

        if (activePage === "rewards") return <RewardMonitor />;

        if (activePage === "experiments") return <ExperimentHistory />;

        if (activePage === "replay") return <ReplayAnalysis />;

        return (
          <PlaceholderPage
            activePage={activePage}
            mode={mode}
          />
        );
      }}
    </AppLayout>
  );
}