import { useState } from "react";
import Sidebar from "../components/Sidebar";
import TopBar from "../components/TopBar";
import { ThemeProvider } from "../context/ThemeContext";

export default function AppLayout({ children }) {
  const [activePage, setActivePage] = useState("overview");

  return (
    <ThemeProvider>
      <div className="st-shell">
        <Sidebar activePage={activePage} setActivePage={setActivePage} />
        <div className="st-main">
          <TopBar />
          <main className="st-content">
            {children({ activePage })}
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}
