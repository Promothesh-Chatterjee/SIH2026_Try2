import { useState } from "react";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";

export default function AppLayout({ children }) {
  const [activePage, setActivePage] = useState("overview");
  const [mode, setMode] = useState("live");

  return (
    <div className="app-shell">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        mode={mode}
        setMode={setMode}
      />

      <div className="app-main">
        <TopBar mode={mode} />

        <main className="app-content">
          {children({
            activePage,
            mode,
          })}
        </main>
      </div>
    </div>
  );
}
