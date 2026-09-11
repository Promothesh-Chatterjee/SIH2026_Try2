import {
  useEffect,
  useState,
} from "react";

import { backend } from "../services/backend";

export default function SystemConnection() {
  const [status, setStatus] =
    useState("CHECKING");

  const [detail, setDetail] =
    useState(
      "Connecting to backend",
    );

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        await backend.api.getSystemStatus();

        if (!active) {
          return;
        }

        setStatus(
          "CONNECTED",
        );

        setDetail(
          "Backend online",
        );
      } catch {
        if (!active) {
          return;
        }

        setStatus(
          "OFFLINE",
        );

        setDetail(
          "Backend unavailable",
        );
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 2500);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const connected = status === "CONNECTED";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "2px 6px",
        background: "var(--panel)",
        border: "1px solid var(--border)",
        borderRadius: 0,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          background: connected ? "var(--success)" : "var(--warning)",
          display: "inline-block",
        }}
      />
      <div>
        <strong style={{ color: "var(--text)" }}>
          {status}
        </strong>
        <span style={{ color: "var(--muted)" }}>
          {" "}
          {detail}
        </span>
      </div>
    </div>
  );
}
