/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        charcoal: {
          DEFAULT: "#0b0e14",
          surface: "#10141d",
          raised: "#151b28",
        },
        navy: "#0f1626",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      boxShadow: {
        "glow-cyan": "0 0 0 1px rgba(34,211,238,0.25), 0 8px 32px -8px rgba(34,211,238,0.25)",
        "glow-soft": "0 8px 30px -12px rgba(0, 0, 0, 0.7)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        flash: {
          "0%": { opacity: "0.65" },
          "100%": { opacity: "0" },
        },
      },
      animation: {
        shimmer: "shimmer 1.8s linear infinite",
        fadeUp: "fadeUp 0.25s ease-out both",
        float: "float 5s ease-in-out infinite",
        flash: "flash 0.5s ease-out forwards",
      },
    },
  },
  plugins: [],
};
