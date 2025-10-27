import type { Config } from "tailwindcss";
import forms from "@tailwindcss/forms";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "#38bdf8",
          foreground: "#020817",
        },
        card: {
          DEFAULT: "#0f172a",
          foreground: "#e2e8f0",
        },
        muted: {
          DEFAULT: "#1e293b",
          foreground: "#cbd5f5",
        },
        border: "#1f2937",
      },
      boxShadow: {
        card: "0 20px 45px -20px rgba(15, 23, 42, 0.65)",
      },
    },
  },
  plugins: [forms],
};

export default config;
