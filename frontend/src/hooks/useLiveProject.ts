import { useEffect, useRef, useState } from "react";
import { api, subscribeEvents } from "../services/api";
import type { JobOut, ProjectDetail } from "../types";

export interface LiveState {
  detail: ProjectDetail | null;
  loading: boolean;
  error: string | null;
  logs: { id: number; ts: string; level: string; message: string }[];
}

/**
 * Loads a project once, then keeps it live through Server-Sent Events.
 * All displayed progress comes from real backend job state - no timers.
 */
export function useLiveProject(projectId: string | undefined): LiveState {
  const [detail, setDetail] = useState<ProjectDetail | null>(null);
  const [logs, setLogs] = useState<LiveState["logs"]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const lastLogId = useRef(0);
  // Logs are scoped to the latest job, so a re-run starts with a clean console
  // instead of appending to (or, past 500 lines, showing only) older runs.
  const jobIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let closed = false;

    const pullLogs = async () => {
      try {
        const res = await api.logs(projectId, lastLogId.current, jobIdRef.current);
        if (res.logs.length) {
          lastLogId.current = res.lastId;
          setLogs((prev) => [...prev, ...res.logs].slice(-500));
        }
      } catch {
        /* transient */
      }
    };

    const refresh = async () => {
      try {
        const d = await api.getProject(projectId);
        if (closed) return;
        const jobId = d.job?.id ?? null;
        if (jobId !== jobIdRef.current) {
          jobIdRef.current = jobId;
          lastLogId.current = 0;
          setLogs([]);
        }
        setDetail(d);
      } catch (e) {
        if (!closed) setError(String(e));
      } finally {
        if (!closed) setLoading(false);
      }
    };

    refresh().then(pullLogs);

    const close = subscribeEvents(projectId, (type, data) => {
      if (type === "log") {
        // events don't carry the log id; refetch logs lazily
        void pullLogs();
        return;
      }
      if (type === "progress") {
        setDetail((prev) =>
          prev && prev.job
            ? {
                ...prev,
                job: {
                  ...prev.job,
                  progress: (data.progress as number) ?? prev.job.progress,
                  stage: (data.message as string) ?? prev.job.stage,
                } as JobOut,
              }
            : prev,
        );
        return;
      }
      if (type === "stage") {
        setDetail((prev) =>
          prev && prev.job ? { ...prev, job: { ...prev.job, stage: data.stage as string } as JobOut } : prev,
        );
        return;
      }
      if (type === "metrics") {
        void refresh();
        return;
      }
      // status / done / error -> full refresh for authoritative state
      void refresh().then(pullLogs);
    });

    // safety net: poll while a job is running even if SSE hiccups
    const interval = setInterval(() => {
      setDetail((prev) => {
        if (prev?.job && (prev.job.status === "running" || prev.job.status === "queued")) {
          void refresh();
          void pullLogs();
        }
        return prev;
      });
    }, 4000);

    return () => {
      closed = true;
      close();
      clearInterval(interval);
    };
  }, [projectId]);

  return { detail, loading, error, logs };
}
