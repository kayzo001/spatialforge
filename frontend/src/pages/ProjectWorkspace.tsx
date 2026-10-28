import { useEffect, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft, Ban, Trash2, Box, Download, RefreshCw,
} from "lucide-react";
import { useLiveProject } from "../hooks/useLiveProject";
import { api } from "../services/api";
import { STAGE_ORDER, STAGE_LABELS, type ArtifactOut } from "../types";
import { StatusBadge } from "./Dashboard";
import ViewerPanel from "../viewers/SceneViewer";
import SplatViewer from "../viewers/SplatViewer";

type Tab = "pipeline" | "viewer" | "analytics";

const LEVEL_COLORS: Record<string, string> = {
  info: "text-slate-400",
  warning: "text-amber-300",
  error: "text-rose-300",
};

function PipelineStages({ stage, status }: { stage: string | null; status: string }) {
  const currentIdx = stage ? STAGE_ORDER.indexOf(stage as (typeof STAGE_ORDER)[number]) : -1;
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {STAGE_ORDER.map((s, i) => {
        const active = i === currentIdx && (status === "running" || status === "queued");
        const done = currentIdx < 0 ? false : i < currentIdx || status === "completed";
        return (
          <div key={s} className="flex items-center gap-1">
            <div
              className={`px-2.5 py-1 rounded text-[11px] border transition-colors ${
                active
                  ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-300"
                  : done
                    ? "border-emerald-600/30 bg-emerald-500/5 text-emerald-400/80"
                    : "border-slate-800 text-slate-600"
              }`}
            >
              {STAGE_LABELS[s]}
            </div>
            {i < STAGE_ORDER.length - 1 && <div className="w-2 h-px bg-slate-800" />}
          </div>
        );
      })}
    </div>
  );
}

function LogConsole({ logs }: { logs: { id: number; ts: string; level: string; message: string }[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [logs.length]);
  return (
    <div
      ref={ref}
      className="h-72 overflow-auto rounded-md bg-[#080b11] border border-slate-800/80 p-3 font-mono text-[11px] leading-relaxed"
    >
      {logs.length === 0 && <div className="text-slate-600">Waiting for pipeline output...</div>}
      {logs.map((l) => (
        <div key={l.id} className={LEVEL_COLORS[l.level] ?? "text-slate-400"}>
          <span className="text-slate-600">{l.ts.slice(11, 19)}</span> {l.message}
        </div>
      ))}
    </div>
  );
}

function artifact(artifacts: ArtifactOut[], kind: string): ArtifactOut | undefined {
  return artifacts.find((a) => a.kind === kind);
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function ViewerTab({ detail }: { detail: NonNullable<ReturnType<typeof useLiveProject>["detail"]> }) {
  const sparse = artifact(detail.artifacts, "sparse_cloud");
  const refined = artifact(detail.artifacts, "refined_cloud");
  const splats = artifact(detail.artifacts, "splats");
  const cameras = artifact(detail.artifacts, "cameras");

  const [mode, setMode] = useState<"sparse" | "refined" | "splats">(
    refined ? "refined" : "sparse",
  );

  const activeUrl =
    mode === "splats" && splats
      ? splats.url
      : mode === "refined" && refined
        ? refined.url
        : sparse?.url ?? null;

  const colmapMetrics = detail.metrics.find((m) => m.kind === "colmap")?.data as any;
  const trainingMetrics = detail.metrics.find((m) => m.kind === "training")?.data as any;

  return (
    <div className="flex gap-4 h-[calc(100vh-13rem)]">
      <div className="flex-1 min-w-0 flex flex-col gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          {sparse && (
            <button
              onClick={() => setMode("sparse")}
              className={`btn-ghost !py-1.5 ${mode === "sparse" ? "!border-cyan-500/50 !text-cyan-300 !shadow-glow-cyan" : ""}`}
            >
              Sparse point cloud
            </button>
          )}
          {refined && (
            <button
              onClick={() => setMode("refined")}
              className={`btn-ghost !py-1.5 ${mode === "refined" ? "!border-cyan-500/50 !text-cyan-300 !shadow-glow-cyan" : ""}`}
            >
              Refined point cloud
            </button>
          )}
          {splats && (
            <button
              onClick={() => setMode("splats")}
              className={`btn-ghost !py-1.5 ${mode === "splats" ? "!border-cyan-500/50 !text-cyan-300 !shadow-glow-cyan" : ""}`}
            >
              Gaussian splats
            </button>
          )}
          {!sparse && !refined && !splats && (
            <div className="text-sm text-slate-500">
              No viewable artifacts yet — run the reconstruction first.
            </div>
          )}
        </div>
        <div className="flex-1 relative min-h-0">
          {mode === "splats" && splats ? (
            <SplatViewer splatUrl={splats.url} background="#0b0e14" />
          ) : (
            <ViewerPanel
              plyUrl={activeUrl}
              camerasUrl={cameras?.url ?? null}
              fileHint={mode === "refined" ? "refined_cloud.ply" : "sparse_cloud.ply"}
            />
          )}
        </div>
      </div>

      {/* Inspector */}
      <div className="w-72 shrink-0 overflow-auto space-y-3">
        <div className="panel p-4">
          <div className="panel-title mb-2">Scene info</div>
          <dl className="space-y-1.5 text-sm">
            <div className="flex justify-between"><dt className="text-slate-500">Method</dt>
              <dd className="text-slate-300">{mode === "splats" ? "Gaussian Splatting" : "COLMAP SfM"}</dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">
              {mode === "splats" ? "Gaussians" : "Points"}</dt>
              <dd className="text-slate-300">
                {mode === "splats"
                  ? (trainingMetrics?.num_gaussians?.toLocaleString?.() ?? "—")
                  : (colmapMetrics?.num_points3d?.toLocaleString() ?? "—")}
              </dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Registered cameras</dt>
              <dd className="text-slate-300">
                {colmapMetrics ? `${colmapMetrics.num_images_registered ?? colmapMetrics.images_registered}/${colmapMetrics.images_uploaded}` : "—"}
              </dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Reproj. error</dt>
              <dd className="text-slate-300">
                {colmapMetrics ? `${Number(colmapMetrics.mean_reprojection_error_px).toFixed(3)} px` : "—"}
              </dd></div>
            <div className="flex justify-between"><dt className="text-slate-500">Training</dt>
              <dd className="text-slate-300 capitalize">{trainingMetrics?.status?.replace("_", " ") ?? "—"}</dd></div>
          </dl>
        </div>
        <div className="panel p-4">
          <div className="panel-title mb-2">Artifacts & exports</div>
          <ul className="space-y-1.5">
            {detail.artifacts
              .filter((a) => a.kind !== "viewer_cloud")
              .map((a) => (
                <li key={`${a.kind}-${a.filename}`}>
                  <a
                    href={a.url}
                    download
                    className="flex items-center justify-between text-xs text-slate-400 hover:text-cyan-300 transition-colors"
                  >
                    <span className="flex items-center gap-1.5">
                      <Box className="w-3 h-3" /> {a.kind}
                      <span className="text-slate-600">.{a.format}</span>
                    </span>
                    <span className="flex items-center gap-1">
                      {fmtSize(a.sizeBytes)} <Download className="w-3 h-3" />
                    </span>
                  </a>
                </li>
              ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, unit }: { label: string; value: string | number | null; unit?: string }) {
  return (
    <div className="panel p-4">
      <div className="text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</div>
      <div className="text-xl font-medium text-slate-100 mt-1">
        {value ?? "Not available"}
        {value != null && unit && <span className="text-sm text-slate-500 ml-1">{unit}</span>}
      </div>
    </div>
  );
}

type Detail = NonNullable<ReturnType<typeof useLiveProject>["detail"]>;

// Metric payloads come from persisted JSON blobs; cast once at the boundary.
/* eslint-disable @typescript-eslint/no-explicit-any */
function AnalyticsBody({ detail }: { detail: Detail }) {
  const colmap = detail.metrics.find((m) => m.kind === "colmap")?.data as any;
  const training = detail.metrics.find((m) => m.kind === "training")?.data as any;
  const evaluation = detail.metrics.find((m) => m.kind === "evaluation")?.data as any;
  const renders = detail.artifacts.filter((a) => a.kind === "render");

  return (
    <div className="space-y-4">
      <div>
        <div className="panel-title mb-2">Reconstruction statistics</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Metric label="Images uploaded" value={colmap?.images_uploaded ?? null} />
          <Metric label="Images registered" value={colmap ? `${colmap.images_registered} (${Math.round((colmap.registration_ratio ?? 0) * 100)}%)` : null} />
          <Metric label="Sparse points" value={colmap?.num_points3d?.toLocaleString() ?? null} />
          <Metric label="Reproj. error" value={colmap ? Number(colmap.mean_reprojection_error_px).toFixed(3) : null} unit="px" />
          <Metric label="Mean track length" value={colmap ? Number(colmap.mean_track_length).toFixed(2) : null} />
          <Metric label="Observations" value={colmap?.num_observations?.toLocaleString() ?? null} />
          <Metric label="Point backend" value={colmap?.point_processing?.backend ?? null} />
          <Metric label="Runtime" value={detail.stats?.duration_s != null ? `${detail.stats.duration_s}s` : null} />
        </div>
      </div>

      <div>
        <div className="panel-title mb-2">Neural training (Gaussian Splatting)</div>
        {training?.status === "completed" ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Metric label="Iterations" value={training.iterations ?? null} />
            <Metric label="Preset" value={training.preset ?? null} />
          </div>
        ) : (
          <div className="panel p-4 text-sm text-slate-400">
            {training?.status === "not_requested" || training?.status === "not_available"
              ? training.reason ?? "Gaussian Splatting training was not run for this project."
              : "No training metrics yet."}
            <div className="text-xs text-slate-500 mt-2">
              Training requires a CUDA GPU (local GPU mode or the bundled Colab notebook).
            </div>
          </div>
        )}
      </div>

      <div>
        <div className="panel-title mb-2">Novel-view evaluation (held-out images)</div>
        {evaluation ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Metric label="PSNR" value={evaluation.psnr != null ? Number(evaluation.psnr).toFixed(2) : null} unit="dB" />
            <Metric label="SSIM" value={evaluation.ssim != null ? Number(evaluation.ssim).toFixed(4) : null} />
            <Metric label="LPIPS" value={evaluation.lpips != null ? Number(evaluation.lpips).toFixed(4) : null} />
          </div>
        ) : (
          <div className="panel p-4 text-sm text-slate-400">
            Not available — evaluation metrics are produced by the GPU training path
            (ns-eval on held-out viewpoints).{renders.length > 0 && " Comparison renders are listed below."}
          </div>
        )}
      </div>

      {renders.length > 0 && (
        <div>
          <div className="panel-title mb-2">Original vs rendered comparison</div>
          <div className="grid grid-cols-2 gap-3">
            {renders.map((r) => (
              <img key={r.url} src={r.url} alt={r.filename} className="rounded border border-slate-800" />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ProjectWorkspace() {
  const { projectId } = useParams();
  const { detail, loading, error, logs } = useLiveProject(projectId);
  const [tab, setTab] = useState<Tab>("pipeline");
  const [busy, setBusy] = useState(false);

  if (loading && !detail) {
    return <div className="p-10 text-sm text-slate-500">Loading project...</div>;
  }
  if (error || !detail) {
    return (
      <div className="p-10 text-sm text-rose-300">
        {error ?? "Project not found."}{" "}
        <Link to="/" className="underline">Back to dashboard</Link>
      </div>
    );
  }

  const job = detail.job;
  const running = job?.status === "running" || job?.status === "queued";

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-8 py-8">
      <div className="flex items-center gap-3">
        <Link to="/" className="btn-ghost !py-1.5 !px-2"><ArrowLeft className="w-4 h-4" /></Link>
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">{detail.name}</h1>
        <StatusBadge status={detail.status} />
        <div className="ml-auto flex items-center gap-2">
          {running && (
            <button className="btn-ghost" disabled={busy}
              onClick={() => act(() => api.cancel(detail.id))}>
              <Ban className="w-4 h-4" /> Cancel
            </button>
          )}
          {detail.status === "completed" && (
            <button className="btn-ghost" disabled={busy}
              onClick={() => act(() => api.reconstruct(detail.id, { mode: detail.mode, preset: detail.preset ?? "balanced" }))}>
              <RefreshCw className="w-4 h-4" /> Re-run
            </button>
          )}
          {!running && (
            <button
              className="btn-ghost !text-rose-300/80 hover:!border-rose-500/40"
              disabled={busy}
              onClick={() => {
                if (confirm(`Delete project "${detail.name}" and all artifacts?`)) {
                  act(() => api.deleteProject(detail.id)).then(() => (window.location.href = "/"));
                }
              }}
            >
              <Trash2 className="w-4 h-4" /> Delete
            </button>
          )}
        </div>
      </div>

      <div className="flex gap-1 mt-5 border-b border-slate-800/70">
        {(["pipeline", "viewer", "analytics"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm capitalize border-b-2 -mb-px transition-colors ${
              tab === t
                ? "border-cyan-400 text-cyan-300"
                : "border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            {t === "viewer" ? "3D Viewer" : t}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tab === "pipeline" && (
          <div className="space-y-4">
            <div className="panel p-5">
              <PipelineStages stage={job?.stage ?? null} status={job?.status ?? detail.status} />
              <div className="mt-4 h-1.5 bg-slate-800 rounded overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    detail.status === "failed"
                      ? "bg-rose-500"
                      : running
                        ? "progress-shimmer animate-shimmer"
                        : detail.status === "completed"
                          ? "bg-emerald-500"
                          : "bg-cyan-500"
                  }`}
                  style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }}
                />
              </div>
              <div className="flex justify-between mt-2 text-xs text-slate-500">
                <span>{Math.round((job?.progress ?? 0) * 100)}%</span>
                {job?.startedAt && (
                  <span>
                    started {new Date(job.startedAt).toLocaleTimeString()}
                    {job.finishedAt && ` · finished ${new Date(job.finishedAt).toLocaleTimeString()}`}
                  </span>
                )}
              </div>
              {detail.error && (
                <div className="mt-3 text-sm text-rose-300 bg-rose-500/5 border border-rose-500/20 rounded p-3">
                  {detail.error}
                </div>
              )}
            </div>
            <div className="panel p-4">
              <div className="panel-title mb-2">Processing logs</div>
              <LogConsole logs={logs} />
            </div>
          </div>
        )}
        {tab === "viewer" && <ViewerTab detail={detail} />}
        {tab === "analytics" && <AnalyticsBody detail={detail} />}
      </div>
    </div>
  );
}
