import React, {
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

    return () => {
      active = false;
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
        background: "#1a1c20",
        border: "1px solid #454653",
        borderRadius: 0,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          background: connected ? "#49df9d" : "#f59e0b",
          display: "inline-block",
        }}
      />
      <div>
        <strong style={{ color: "#e2e2e8" }}>
          {status}
        </strong>
        <span style={{ color: "#908f9e" }}>
          {" "}
          {detail}
        </span>
      </div>
    </div>
  );
}
