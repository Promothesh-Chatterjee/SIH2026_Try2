import AppLayout from "./layouts/AppLayout";
import MissionOverview from "./pages/MissionOverview";
import BenchmarkPage from "./pages/BenchmarkPage";
import LiveSpectrum from "./pages/LiveSpectrum";
import SmartScan from "./pages/SmartScan";
import Receiver from "./pages/Receiver";
import Interception from "./pages/Interception";
import Emitters from "./pages/Emitters";
import Performance from "./pages/Performance";
import DatasetAudit from "./pages/DatasetAudit";
import SystemConfig from "./pages/SystemConfig";
import "./App.css";

const PAGES = {
  overview: MissionOverview,
  benchmark: BenchmarkPage,
  spectrum: LiveSpectrum,
  "smart-scan": SmartScan,
  receiver: Receiver,
  interception: Interception,
  emitters: Emitters,
  performance: Performance,
  dataset: DatasetAudit,
  system: SystemConfig,
};

export default function App() {
  return (
    <AppLayout>
      {({ activePage }) => {
        const Page = PAGES[activePage] ?? MissionOverview;
        return <Page />;
      }}
    </AppLayout>
  );
}
