import { Link, Outlet, useLocation } from "react-router-dom";
import { Boxes, Cpu, Plus, LayoutDashboard, Github } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { SystemCapabilities } from "../types";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/new", label: "New Reconstruction", icon: Plus },
];

export default function Layout() {
  const location = useLocation();
  const [caps, setCaps] = useState<SystemCapabilities | null>(null);

  useEffect(() => {
    api.capabilities().then(setCaps).catch(() => setCaps(null));
  }, [location.pathname]);

  const modeLabel = caps
    ? caps.modes.local_gpu_training
      ? "GPU ready"
      : caps.modes.cpu_reconstruction
        ? "CPU mode"
        : "Setup needed"
    : "Detecting...";
  const modeActive = caps?.modes.cpu_reconstruction || caps?.modes.local_gpu_training;

  return (
    <div className="h-full flex bg-[#0b0e14]">
      <aside className="w-16 lg:w-60 shrink-0 transition-[width] border-r border-slate-800/70 flex flex-col bg-gradient-to-b from-[#0d1119] via-[#0b0e14] to-[#0a0d13]">
        <Link to="/" className="flex items-center gap-3 px-3.5 lg:px-5 h-16 border-b border-slate-800/70 group">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-400 to-indigo-600 flex items-center justify-center shadow-glow-cyan transition-transform group-hover:scale-105">
            <Boxes className="w-5 h-5 text-[#04141a]" strokeWidth={2} />
          </div>
          <div className="hidden lg:block">
            <div className="font-semibold tracking-tight text-slate-100 leading-tight">
              Spatial<span className="gradient-text">Forge</span>
            </div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
              photos → 3D worlds
            </div>
          </div>
        </Link>

        <div className="hidden lg:block px-5 pt-5 pb-1 text-[10px] uppercase tracking-[0.18em] text-slate-600">
          Workspace
        </div>
        <nav className="px-3 pt-4 lg:pt-0 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active =
              to === "/" ? location.pathname === "/" : location.pathname.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                title={label}
                className={`relative flex items-center justify-center lg:justify-start gap-2.5 px-3 py-2 rounded-md text-sm transition-all ${
                  active
                    ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/25 shadow-glow-cyan"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40 border border-transparent active:scale-[0.98]"
                }`}
              >
                {active && (
                  <span className="absolute -left-3 top-1/2 -translate-y-1/2 h-5 w-0.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.9)]" />
                )}
                <Icon className="w-4 h-4 shrink-0" strokeWidth={1.75} />
                <span className="hidden lg:inline">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto p-2.5 lg:p-4 border-t border-slate-800/70 space-y-3">
          <div
            title={modeLabel}
            className={`flex items-center justify-center lg:justify-start gap-2 text-xs rounded-md border px-2.5 py-2 ${
              modeActive
                ? "border-emerald-700/40 bg-emerald-500/5 text-emerald-300/90"
                : "border-amber-700/40 bg-amber-500/5 text-amber-300/90"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">{modeLabel}</span>
            {modeActive && (
              <span className="hidden lg:block ml-auto w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            )}
          </div>
          {caps && (
            <div className="hidden lg:block text-[11px] text-slate-600 leading-relaxed">
              {caps.platform.system} · {caps.hardware.cpu_cores ?? "?"} cores ·{" "}
              {caps.hardware.ram_gb ?? "?"} GB RAM
              {caps.hardware.gpus.length > 0 &&
                ` · ${caps.hardware.gpus[0].name.replace("NVIDIA GeForce ", "")}`}
            </div>
          )}
          <div className="hidden lg:flex items-center gap-1.5 text-[10px] text-slate-600">
            <Github className="w-3 h-3" />
            SpatialForge v0.1.0
          </div>
        </div>
      </aside>
      <main className="flex-1 min-w-0 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
