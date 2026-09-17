import { useTheme } from "../context/ThemeContext";

export default function ThemeToggle({ compact = false, showLabel = true, className = "" }) {
  const { isDark, toggleTheme } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className={`st-theme-toggle ${className}`}
      title={
        isDark
          ? "Switch to Light Mode (white background)"
          : "Switch to Dark Mode (black background)"
      }
      aria-label={
        isDark
          ? "Switch to Light Mode (white background)"
          : "Switch to Dark Mode (black background)"
      }
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: compact ? "2px 6px" : "3px 8px",
        background: "var(--panel)",
        border: "1px solid var(--border)",
        borderRadius: 0,
        color: isDark ? "#fef08a" : "#4338ca",
        cursor: "pointer",
        userSelect: "none",
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
        fontSize: compact ? "10px" : "11px",
        fontWeight: 600,
        letterSpacing: "0.04em",
        whiteSpace: "nowrap",
        transition: "background 0.15s ease, border-color 0.15s ease, color 0.15s ease",
      }}
    >
      <span
        className="material-symbols-outlined"
        style={{
          fontSize: compact ? 14 : 16,
          lineHeight: 1,
          display: "inline-block",
        }}
      >
        {isDark ? "light_mode" : "dark_mode"}
      </span>
      {showLabel && (
        <span style={{ color: "var(--text)" }}>
          {isDark ? "LIGHT MODE" : "DARK MODE"}
        </span>
      )}
      <span
        style={{
          width: 6,
          height: 6,
          background: isDark ? "#49df9d" : "#0284c7",
          display: "inline-block",
          marginLeft: 2,
        }}
      />
    </button>
  );
}
