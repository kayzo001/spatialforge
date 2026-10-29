import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { UploadCloud, Check, X, Loader2, Zap, Scale, Gem } from "lucide-react";
import { api } from "../services/api";
import type { ProjectOut, SystemCapabilities, UploadReport } from "../types";

const PRESETS = [
  { key: "fast_preview", label: "Fast Preview", icon: Zap,
    desc: "Quick iteration pass, lower GPU cost (GPU training only)." },
  { key: "balanced", label: "Balanced", icon: Scale,
    desc: "Balance of speed, geometry and appearance quality." },
  { key: "high_quality", label: "High Quality", icon: Gem,
    desc: "Prioritizes reconstruction detail; slowest option." },
];

export default function NewProject() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [project, setProject] = useState<ProjectOut | null>(null);
  const [report, setReport] = useState<UploadReport | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [preset, setPreset] = useState("balanced");
  const [mode, setMode] = useState<"cpu" | "gpu" | "colab">("cpu");
  const [caps, setCaps] = useState<SystemCapabilities | null>(null);
  const [capsLoading, setCapsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refreshCaps = useCallback(() => {
    setCapsLoading(true);
    api
      .capabilities()
      .then(setCaps)
      .catch(() => setCaps(null))
      .finally(() => setCapsLoading(false));
  }, []);

  // Detect COLMAP / GPU support as soon as the page opens. Without this the
  // page showed "COLMAP was not detected" until Re-detect was clicked.
  useEffect(() => {
    refreshCaps();
  }, [refreshCaps]);

  const uploadFiles = useCallback(
    (list: File[]) => {
      if (!list.length) return;
      setError(null);
      setUploading(true);
      setProgress(0);
      const run = async () => {
        let proj = project;
        if (!proj) {
          proj = await api.createProject({
            name: name.trim() || `Reconstruction ${new Date().toLocaleDateString()}`,
          });
          setProject(proj);
        }
        // upload in batches to show progress and stay under request limits
        const batch = 25;
        let last: UploadReport | null = null;
        for (let i = 0; i < list.length; i += batch) {
          const slice = list.slice(i, i + batch);
          last = await new Promise<UploadReport>((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open("POST", `/api/projects/${proj!.id}/images`);
            xhr.upload.onprogress = (e) =>
              setProgress((i + e.loaded / Math.max(1, e.total) * batch) / list.length);
            xhr.onload = () => {
              if (xhr.status === 200) resolve(JSON.parse(xhr.responseText));
              else reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
            };
            xhr.onerror = () => reject(new Error("Network error during upload"));
            const form = new FormData();
            slice.forEach((f) => form.append("files", f));
            xhr.send(form);
          });
        }
        setReport(last);
        setUploading(false);
      };
      run().catch((e) => {
        setError(String(e));
        setUploading(false);
      });
    },
    [name, project],
  );

  const start = async () => {
    if (!project || !report) return;
    setStarting(true);
    try {
      await api.reconstruct(project.id, { preset, mode, eval_fraction: 0.1 });
      navigate(`/projects/${project.id}`);
    } catch (e) {
      setError(String(e));
      setStarting(false);
    }
  };

  const cpuReady = caps?.modes.cpu_reconstruction ?? false;
  const gpuReady = caps?.modes.local_gpu_training ?? false;

  return (
    <div className="max-w-3xl mx-auto px-8 py-10">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-100">New Reconstruction</h1>
      <p className="text-sm text-slate-500 mt-1">
        Photograph one object or scene from many angles with strong overlap.
        20-80 sharp, textured images work best.
      </p>

      <div className="panel p-5 mt-6">
        <label className="panel-title block mb-2">Project name</label>
        <input
          className="input"
          placeholder="e.g. Ceramic vase"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={!!project}
        />
      </div>

      <div
        className={`panel mt-4 p-10 border-dashed text-center cursor-pointer transition-colors ${
          dragOver ? "border-cyan-500/60 bg-cyan-500/5" : "border-slate-700/70"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          uploadFiles(Array.from(e.dataTransfer.files));
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp,application/zip,.zip"
          className="hidden"
          onChange={(e) => uploadFiles(Array.from(e.target.files ?? []))}
        />
        {uploading ? (
          <div>
            <Loader2 className="w-8 h-8 text-cyan-400 mx-auto animate-spin" />
            <p className="mt-3 text-sm text-slate-300">Uploading and validating images...</p>
            <div className="h-1.5 w-64 bg-slate-800 rounded mx-auto mt-3 overflow-hidden">
              <div
                className="h-full bg-cyan-500 transition-all"
                style={{ width: `${Math.round(progress * 100)}%` }}
              />
            </div>
          </div>
        ) : (
          <div>
            <UploadCloud className="w-8 h-8 text-slate-500 mx-auto" />
            <p className="mt-3 text-sm text-slate-300">
              Drop images or a .zip archive here, or click to browse
            </p>
            <p className="text-xs text-slate-500 mt-1">
              JPEG · PNG · WebP · ZIP — up to 300 images, arbitrary order
            </p>
          </div>
        )}
      </div>

      {report && (
        <div className="panel p-5 mt-4">
          <div className="panel-title mb-3">Dataset validation</div>
          <div className="flex items-center gap-2 text-sm text-emerald-300">
            <Check className="w-4 h-4" /> {report.added.length} image(s) accepted
          </div>
          {report.rejected.length > 0 && (
            <div className="text-sm text-rose-300 mt-2">
              <X className="w-4 h-4 inline" /> {report.rejected.length} rejected:{" "}
              {report.rejected.slice(0, 3).map((r) => r.filename).join(", ")}
              {report.rejected.length > 3 && " …"}
            </div>
          )}
          {report.summary.warnings.map((w, i) => (
            <div key={i} className="text-xs text-amber-300/90 mt-2">⚠ {w}</div>
          ))}
        </div>
      )}

      <div className="panel p-5 mt-4">
        <div className="panel-title mb-3">Training preset</div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {PRESETS.map(({ key, label, icon: Icon, desc }) => (
            <button
              key={key}
              onClick={() => setPreset(key)}
              className={`text-left rounded-lg border p-3 transition-colors ${
                preset === key
                  ? "border-cyan-500/50 bg-cyan-500/5"
                  : "border-slate-800 hover:border-slate-600"
              }`}
            >
              <Icon className="w-4 h-4 text-cyan-400" />
              <div className="text-sm text-slate-200 mt-1.5">{label}</div>
              <div className="text-[11px] text-slate-500 mt-1 leading-snug">{desc}</div>
            </button>
          ))}
        </div>

        <div className="panel-title mt-5 mb-3">Execution mode</div>
        <div className="space-y-2">
          {[
            { key: "cpu", label: "CPU reconstruction", desc: "COLMAP SfM on this machine. Point cloud + camera poses. No neural training.", enabled: cpuReady },
            { key: "gpu", label: "Local GPU training", desc: "Gaussian Splatting via Nerfstudio (CUDA required).", enabled: gpuReady },
            { key: "colab", label: "Google Colab", desc: "Run the bundled notebook on Colab's GPU and import the artifacts here.", enabled: true },
          ].map(({ key, label, desc, enabled }) => (
            <label
              key={key}
              className={`flex items-start gap-3 rounded-lg border p-3 transition-colors ${
                capsLoading && key !== "colab"
                  ? "opacity-60 cursor-wait border-slate-800"
                  : !enabled
                  ? "opacity-40 cursor-not-allowed border-slate-800"
                  : mode === key
                    ? "border-cyan-500/50 bg-cyan-500/5 cursor-pointer"
                    : "border-slate-800 hover:border-slate-600 cursor-pointer"
              }`}
            >
              <input
                type="radio"
                name="mode"
                className="mt-1 accent-cyan-500"
                checked={mode === key}
                disabled={!enabled || (capsLoading && key !== "colab")}
                onChange={() => setMode(key as "cpu" | "gpu" | "colab")}
              />
              <div>
                <div className="text-sm text-slate-200">
                  {label}
                  {capsLoading && key !== "colab" ? (
                    <span className="ml-2 text-[10px] uppercase tracking-wider text-slate-500">
                      <Loader2 className="w-3 h-3 inline animate-spin mr-1" />
                      checking…
                    </span>
                  ) : (
                    !enabled && (
                      <span className="ml-2 text-[10px] uppercase tracking-wider text-amber-400/80">
                        {key === "cpu" ? "COLMAP missing" : "not available locally"}
                      </span>
                    )
                  )}
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">{desc}</div>
              </div>
            </label>
          ))}
        </div>
        {!capsLoading && !cpuReady && (
          <p className="text-xs text-amber-300/90 mt-3">
            COLMAP was not detected on this machine — run{" "}
            <code className="text-amber-200">python scripts/install_colmap.py</code> to enable
            reconstruction, then refresh.
            <button onClick={refreshCaps} className="btn-ghost ml-2 !py-1 !px-2 text-xs">
              Re-detect
            </button>
          </p>
        )}
      </div>

      {error && (
        <div className="panel p-4 mt-4 text-sm text-rose-300">{error}</div>
      )}

      <div className="flex justify-end mt-6">
        <button className="btn-primary" onClick={start} disabled={!report || starting}>
          {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
          Start reconstruction
        </button>
      </div>
    </div>
  );
}
