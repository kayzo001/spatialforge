import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Plus, CircleAlert, ArrowUpRight, ImageOff } from "lucide-react";
import { api } from "../services/api";
import type { ProjectOut } from "../types";

const STATUS_STYLES: Record<string, string> = {
  draft: "bg-slate-700/40 text-slate-300 border-slate-600/50",
  queued: "bg-amber-500/10 text-amber-300 border-amber-500/25",
  processing: "bg-cyan-500/10 text-cyan-300 border-cyan-500/25",
  completed: "bg-emerald-500/10 text-emerald-300 border-emerald-500/25",
  failed: "bg-rose-500/10 text-rose-300 border-rose-500/25",
  canceled: "bg-slate-600/30 text-slate-400 border-slate-600/40",
};

const STATUS_ACCENT: Record<string, string> = {
  draft: "from-slate-600/60",
  queued: "from-amber-400/70",
  processing: "from-cyan-400/80",
  completed: "from-emerald-400/70",
  failed: "from-rose-400/70",
  canceled: "from-slate-600/40",
};

export function StatusBadge({ status }: { status: string }) {
  const live = status === "processing" || status === "queued";
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[11px] font-medium capitalize ${
        STATUS_STYLES[status] ?? STATUS_STYLES.draft
      }`}
    >
      {live && (
        <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      )}
      {status}
    </span>
  );
}

function WireCube({ className }: { className?: string }) {
  // decorative isometric wireframe cube for the empty state
  return (
    <svg viewBox="0 0 120 120" fill="none" className={className}>
      <g stroke="currentColor" strokeWidth="1.2" opacity="0.9">
        <path d="M60 12 104 36 104 84 60 108 16 84 16 36 Z" />
        <path d="M60 12 60 60 M104 36 60 60 16 36 M60 60 60 108" opacity="0.6" />
        <path d="M104 84 60 60 16 84" opacity="0.35" strokeDasharray="3 4" />
      </g>
      <g fill="#22d3ee">
        <circle cx="60" cy="12" r="2.5" />
        <circle cx="104" cy="36" r="2.5" />
        <circle cx="16" cy="36" r="2.5" />
        <circle cx="60" cy="108" r="2.5" />
      </g>
    </svg>
  );
}

export default function Dashboard() {
  const [projects, setProjects] = useState<ProjectOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-8 py-10">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            <span className="text-slate-100">Reconstruction</span>{" "}
            <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Structure-from-Motion reconstructions and neural scene renders.
          </p>
        </div>
        <Link to="/new" className="btn-primary">
          <Plus className="w-4 h-4" /> New Reconstruction
        </Link>
      </div>

      {error && (
        <div className="panel p-4 text-sm text-rose-300 flex items-center gap-2">
          <CircleAlert className="w-4 h-4" /> Could not reach the backend: {error}
        </div>
      )}

      {projects && projects.length === 0 && (
        <div className="panel p-12 text-center relative overflow-hidden">
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
          <WireCube className="w-24 h-24 mx-auto text-cyan-500/70 animate-float" />
          <p className="text-slate-200 font-medium mt-4">No reconstructions yet</p>
          <p className="text-sm text-slate-500 mt-2 max-w-md mx-auto leading-relaxed">
            Upload 20–80 overlapping photos of an object or scene. SpatialForge
            estimates camera poses, reconstructs sparse geometry, and optionally
            trains a Gaussian Splat scene.
          </p>
          <Link to="/new" className="btn-primary mt-6">
            <Plus className="w-4 h-4" /> Create your first reconstruction
          </Link>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {(projects ?? []).map((p) => (
          <Link
            key={p.id}
            to={`/projects/${p.id}`}
            className="panel panel-hover group relative overflow-hidden flex flex-col"
          >
            <div
              className={`absolute inset-x-0 top-0 h-0.5 z-10 bg-gradient-to-r ${
                STATUS_ACCENT[p.status] ?? STATUS_ACCENT.draft
              } to-transparent`}
            />
            <div className="relative h-36 bg-[#0d1119] border-b border-slate-800/60 overflow-hidden">
              {p.thumbnailUrl ? (
                <img
                  src={p.thumbnailUrl}
                  alt=""
                  loading="lazy"
                  className="w-full h-full object-cover opacity-80 group-hover:opacity-100 group-hover:scale-[1.03] transition-all duration-300"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-700">
                  <ImageOff className="w-6 h-6" />
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-[#0b0e14]/80 via-transparent to-transparent" />
              <div className="absolute top-2.5 right-2.5 rounded bg-[#0b0e14]/85 backdrop-blur-sm">
                <StatusBadge status={p.status} />
              </div>
            </div>
            <div className="p-5 pt-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="font-medium text-slate-200 line-clamp-2 break-words group-hover:text-cyan-300 transition-colors">
                  {p.name}
                  <ArrowUpRight className="w-3.5 h-3.5 inline ml-1 align-[-2px] opacity-0 group-hover:opacity-100 transition-all" />
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  {new Date(p.createdAt).toLocaleString()}
                  {p.isDemo && " · demo"}
                </div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center">
              <div className="rounded bg-[#0d1119] border border-slate-800/60 py-2 hover:border-cyan-900/60 transition-colors">
                <div className="text-sm font-medium text-slate-200">{p.imageCount}</div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500">images</div>
              </div>
              <div className="rounded bg-[#0d1119] border border-slate-800/60 py-2 hover:border-cyan-900/60 transition-colors">
                <div className="text-sm font-medium text-slate-200">
                  {p.stats?.num_points3d?.toLocaleString() ?? "—"}
                </div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500">points</div>
              </div>
              <div className="rounded bg-[#0d1119] border border-slate-800/60 py-2 hover:border-cyan-900/60 transition-colors">
                <div className="text-sm font-medium text-slate-200">
                  {p.stats?.duration_s != null ? `${Math.round(p.stats.duration_s)}s` : "—"}
                </div>
                <div className="text-[10px] uppercase tracking-wider text-slate-500">runtime</div>
              </div>
            </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
