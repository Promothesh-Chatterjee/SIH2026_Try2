import { useState } from "react";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";

export default function AppLayout({ children }) {
  const [activePage, setActivePage] = useState("overview");
  const [mode, setMode] = useState("live");

  return (
    <div className="st-shell">
      <Sidebar activePage={activePage} setActivePage={setActivePage} mode={mode} setMode={setMode} />
      <div className="st-main">
        <TopBar mode={mode} />
        <main className="st-content">
          {children({ activePage, mode })}
        </main>
      </div>
    </div>
  );
}
