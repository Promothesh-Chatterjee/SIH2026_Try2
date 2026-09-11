/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext({
  theme: "dark",
  isDark: true,
  toggleTheme: () => {},
  setTheme: () => {},
});

const THEME_STORAGE_KEY = "smartscan_theme";

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(() => {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved === "light" || saved === "dark") {
        return saved;
      }
    } catch {
      // Ignore local storage errors
    }
    return "dark";
  });

  const setTheme = (newTheme) => {
    const validTheme = newTheme === "light" ? "light" : "dark";
    setThemeState(validTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, validTheme);
    } catch {
      // Ignore local storage errors
    }
  };

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "light") {
      root.classList.remove("dark");
      root.classList.add("light");
      root.setAttribute("data-theme", "light");
      root.style.colorScheme = "light";
      if (document.body) {
        document.body.classList.remove("dark");
        document.body.classList.add("light");
        document.body.style.backgroundColor = "#ffffff";
        document.body.style.color = "#0f172a";
      }
    } else {
      root.classList.remove("light");
      root.classList.add("dark");
      root.setAttribute("data-theme", "dark");
      root.style.colorScheme = "dark";
      if (document.body) {
        document.body.classList.remove("light");
        document.body.classList.add("dark");
        document.body.style.backgroundColor = "#111317";
        document.body.style.color = "#e2e2e8";
      }
    }
  }, [theme]);

  const value = {
    theme,
    isDark: theme === "dark",
    toggleTheme,
    setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
