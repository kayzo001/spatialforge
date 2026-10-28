import type {
  JobOut,
  MetricsBlock,
  ProjectDetail,
  ProjectOut,
  SystemCapabilities,
} from "../types";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  // system -----------------------------------------------------------------
  health: () =>
    fetch("/api/health").then((r) => handle<{ status: string }>(r)),
  capabilities: (refresh = false) =>
    fetch(`/api/system/capabilities${refresh ? "?refresh=true" : ""}`).then((r) =>
      handle<SystemCapabilities>(r),
    ),
  stages: () =>
    fetch("/api/system/stages").then(
      (r) => handle<{ id: string; label: string }[]>(r),
    ),

  // projects ---------------------------------------------------------------
  listProjects: () =>
    fetch("/api/projects").then((r) => handle<ProjectOut[]>(r)),
  getProject: (id: string) =>
    fetch(`/api/projects/${id}`).then((r) => handle<ProjectDetail>(r)),
  createProject: (body: { name: string; description?: string; mode?: string; preset?: string }) =>
    fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handle<ProjectOut>(r)),
  deleteProject: (id: string) =>
    fetch(`/api/projects/${id}`, { method: "DELETE" }).then((r) =>
      handle<void>(r),
    ),

  // uploads ----------------------------------------------------------------
  uploadImages: (id: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return fetch(`/api/projects/${id}/images`, {
      method: "POST",
      body: form,
    }).then((r) =>
      handle<{
        added: { id: string; filename: string; quality: Record<string, unknown> }[];
        rejected: { filename: string; reason: string }[];
        errors: string[];
        summary: { image_count: number; warnings: string[] };
      }>(r),
    );
  },

  // jobs -------------------------------------------------------------------
  reconstruct: (id: string, body: { preset?: string; mode?: string; eval_fraction?: number }) =>
    fetch(`/api/projects/${id}/reconstruct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handle<{ jobId: string; status: string }>(r)),
  cancel: (id: string) =>
    fetch(`/api/projects/${id}/cancel`, { method: "POST" }).then((r) =>
      handle<{ cancelRequested: boolean }>(r),
    ),
  status: (id: string) =>
    fetch(`/api/projects/${id}/status`).then(
      (r) => handle<{ project: ProjectOut; job: JobOut | null }>(r),
    ),
  logs: (id: string, afterId = 0) =>
    fetch(`/api/projects/${id}/logs?after_id=${afterId}`).then((r) =>
      handle<{ logs: { id: number; ts: string; level: string; message: string }[]; lastId: number }>(r),
    ),
  metrics: (id: string) =>
    fetch(`/api/projects/${id}/metrics`).then((r) => handle<MetricsBlock[]>(r)),
};

/** Server-Sent Events subscription for live job updates. */
export function subscribeEvents(
  projectId: string,
  onEvent: (type: string, data: Record<string, unknown>) => void,
): () => void {
  const es = new EventSource(`/api/projects/${projectId}/events`);
  es.onmessage = (msg) => {
    try {
      const parsed = JSON.parse(msg.data);
      onEvent(parsed.type, parsed.data ?? {});
    } catch {
      /* ignore malformed frames */
    }
  };
  return () => es.close();
}
