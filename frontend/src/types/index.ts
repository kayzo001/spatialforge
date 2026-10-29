// Mirrors backend/app/schemas.py — keep both sides in sync.
export type ProjectStatus =
  | "draft"
  | "queued"
  | "processing"
  | "completed"
  | "failed"
  | "canceled";

export type ExecutionMode = "cpu" | "gpu" | "colab";
export type Preset = "fast_preview" | "balanced" | "high_quality";

export interface ImageQuality {
  blur_score?: number | null;
  is_likely_blurry?: boolean;
  is_duplicate_of?: string | null;
  too_dark?: boolean;
  warnings?: string[];
}

export interface ProjectImage {
  id: string;
  filename: string;
  width: number | null;
  height: number | null;
  fileSize?: number | null;
  split: string;
  quality: ImageQuality;
  url: string;
}

export interface ProjectStats {
  images_registered?: number;
  images_uploaded?: number;
  num_points3d?: number;
  mean_reprojection_error_px?: number;
  duration_s?: number;
}

export interface JobOut {
  id: string;
  projectId: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "canceled";
  stage: string | null;
  progress: number;
  error: string | null;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface ArtifactOut {
  kind: string;
  filename: string;
  format: string;
  sizeBytes: number;
  url: string;
}

export interface MetricsBlock {
  kind: string;
  data: Record<string, unknown>;
  createdAt: string;
}

export interface ProjectOut {
  id: string;
  name: string;
  description: string;
  status: ProjectStatus;
  mode: ExecutionMode;
  preset: Preset | null;
  error: string | null;
  isDemo: boolean;
  imageCount: number;
  thumbnailUrl?: string | null;
  stats: ProjectStats | null;
  createdAt: string;
  updatedAt: string;
}

export interface ProjectDetail extends ProjectOut {
  images: ProjectImage[];
  job: JobOut | null;
  artifacts: ArtifactOut[];
  metrics: MetricsBlock[];
  availableStages: string[];
}

export interface SystemCapabilities {
  platform: { system: string; release: string; python: string };
  hardware: {
    cpu_model: string | null;
    cpu_cores: number | null;
    ram_gb: number | null;
    gpus: { name: string; vram_mb: number; compute_capability: string }[];
    disk_free_gb: number | null;
  };
  dependencies: {
    colmap: { available: boolean; version: string | null; path: string | null };
    open3d: { available: boolean; version: string | null };
    torch: { available: boolean; version: string | null; cuda_available: boolean };
    nerfstudio: { available: boolean };
    lpips: { available: boolean };
  };
  modes: {
    cpu_reconstruction: boolean;
    local_gpu_training: boolean;
    colab_gpu_training: boolean;
  };
  notes: string[];
}

export interface UploadReport {
  added: { id: string; filename: string; quality: ImageQuality }[];
  rejected: { filename: string; reason: string }[];
  errors: string[];
  summary: {
    image_count: number;
    blurry: string[];
    duplicates: string[];
    warnings: string[];
  };
}

export const STAGE_ORDER = [
  "upload",
  "validation",
  "preprocessing",
  "feature_extraction",
  "matching",
  "sparse_reconstruction",
  "point_processing",
  "training",
  "evaluation",
  "export",
] as const;

export const STAGE_LABELS: Record<string, string> = {
  upload: "Upload",
  validation: "Validation",
  preprocessing: "Preprocessing",
  feature_extraction: "Feature Extraction",
  matching: "Matching",
  sparse_reconstruction: "Sparse Reconstruction",
  point_processing: "Point Processing",
  training: "Neural Training",
  evaluation: "Evaluation",
  export: "Export",
};
