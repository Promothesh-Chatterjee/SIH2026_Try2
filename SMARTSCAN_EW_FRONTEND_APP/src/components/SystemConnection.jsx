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

  return (
    <div
      className={`system-connection ${status.toLowerCase()}`}
    >
      <span className="connection-dot" />

      <div>
        <strong>
          {status}
        </strong>

        <span>
          {detail}
        </span>
      </div>
    </div>
  );
}