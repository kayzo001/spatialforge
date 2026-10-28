import { useEffect, useRef } from "react";
import { Viewer as GSViewer } from "@mkkellogg/gaussian-splats-3d";

interface SplatViewerProps {
  splatUrl: string;
  background: string;
}

/**
 * True anisotropic 3D Gaussian Splat rendering via @mkkellogg/gaussian-splats-3d.
 * Only mounted when the project actually has a trained splat artifact.
 */
export default function SplatViewer({ splatUrl, background }: SplatViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<GSViewer | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const viewer = new GSViewer({
      rootElement: container,
      dynamicScene: false,
      sharedMemoryForWorkers: false,
    });
    viewerRef.current = viewer;

    let disposed = false;
    viewer
      .addSplatScene(splatUrl, {
        splatAlphaRemovalThreshold: 5,
        showLoadingUI: true,
        position: [0, 1, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
      })
      .then(() => {
        if (disposed) return;
        viewer.start();
      })
      .catch(() => {
        /* container may unmount during load */
      });

    return () => {
      disposed = true;
      try {
        viewer.stop();
        viewer.dispose();
      } catch {
        /* already disposed */
      }
      viewerRef.current = null;
    };
  }, [splatUrl]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full"
      style={{ background }}
    />
  );
}
