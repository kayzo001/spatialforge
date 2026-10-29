import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  Camera, Crosshair, Grid3x3, Maximize2, Play, Pause, Download, Keyboard,
} from "lucide-react";
import { cloudGeometry, parseBinaryPly, type PlyCloud } from "./plyLoader";
import type { CameraPose } from "./types";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type ColorMode = "original" | "cyan" | "depth";

interface FlyTarget {
  pos: THREE.Vector3;
  quat: THREE.Quaternion;
  target: THREE.Vector3;
}

const smoothstep = (t: number) => t * t * (3 - 2 * t);

/**
 * cameras.json stores c2w as nested rows (row-major). THREE.Matrix4.fromArray
 * expects column-major, so it would silently transpose the pose: frustums turn
 * into long projective spikes and every camera lands at the origin.
 * Matrix4.set() takes row-major arguments, which matches the file.
 */
function poseMatrix(c2w: number[][]): THREE.Matrix4 {
  const r = c2w.flat();
  return new THREE.Matrix4().set(
    r[0], r[1], r[2], r[3],
    r[4], r[5], r[6], r[7],
    r[8], r[9], r[10], r[11],
    r[12] ?? 0, r[13] ?? 0, r[14] ?? 0, r[15] ?? 1,
  );
}

/** Radial gradient coloring from the cloud centroid: indigo core -> cyan -> amber rim. */
function depthColorArray(ply: PlyCloud): Float32Array {
  const n = ply.vertexCount;
  const out = new Float32Array(n * 3);
  const inner = new THREE.Color("#1e3a8a");
  const mid = new THREE.Color("#22d3ee");
  const rim = new THREE.Color("#fbbf24");
  const tmp = new THREE.Color();
  const p = ply.positions;
  const centroid = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    centroid[0] += p[i * 3];
    centroid[1] += p[i * 3 + 1];
    centroid[2] += p[i * 3 + 2];
  }
  centroid[0] /= n;
  centroid[1] /= n;
  centroid[2] /= n;
  let minD = Infinity;
  let maxD = 0;
  const dists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = p[i * 3] - centroid[0];
    const dy = p[i * 3 + 1] - centroid[1];
    const dz = p[i * 3 + 2] - centroid[2];
    dists[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dists[i] < minD) minD = dists[i];
    if (dists[i] > maxD) maxD = dists[i];
  }
  const range = Math.max(maxD - minD, 1e-9);
  for (let i = 0; i < n; i++) {
    const t = (dists[i] - minD) / range;
    if (t < 0.6) tmp.lerpColors(inner, mid, t / 0.6);
    else tmp.lerpColors(mid, rim, (t - 0.6) / 0.4);
    out[i * 3] = tmp.r;
    out[i * 3 + 1] = tmp.g;
    out[i * 3 + 2] = tmp.b;
  }
  return out;
}

type OrbitControlsRef = React.MutableRefObject<React.ComponentRef<typeof OrbitControls> | null>;

// ---------------------------------------------------------------------------
// in-canvas components
// ---------------------------------------------------------------------------

function PointCloud({
  ply,
  colorMode,
  pointScale,
  sceneRadius,
  onHoverPoint,
}: {
  ply: PlyCloud;
  colorMode: ColorMode;
  pointScale: number;
  sceneRadius: number;
  onHoverPoint: (info: { index: number; xyz: [number, number, number] } | null) => void;
}) {
  const geometry = useMemo(() => cloudGeometry(ply), [ply]);
  const depthColors = useMemo(
    () => (ply.vertexCount <= 2_000_000 ? depthColorArray(ply) : null),
    [ply],
  );

  useEffect(() => {
    if (colorMode === "original") {
      geometry.setAttribute("color", new THREE.BufferAttribute(ply.colors, 3));
    } else if (colorMode === "depth" && depthColors) {
      geometry.setAttribute("color", new THREE.BufferAttribute(depthColors, 3));
    }
    if (geometry.attributes.color) geometry.attributes.color.needsUpdate = true;
  }, [geometry, colorMode, ply, depthColors]);

  const hoverEnabled = ply.vertexCount <= 300_000;

  return (
    <points
      geometry={geometry}
      onPointerMove={(e) => {
        if (!hoverEnabled) return;
        const idx = e.index ?? -1;
        if (idx < 0) return;
        onHoverPoint({
          index: idx,
          xyz: [ply.positions[idx * 3], ply.positions[idx * 3 + 1], ply.positions[idx * 3 + 2]],
        });
      }}
      onPointerOut={() => onHoverPoint(null)}
    >
      <pointsMaterial
        size={Math.max((sceneRadius / 300) * pointScale, 0.004)}
        vertexColors={colorMode !== "cyan"}
        color={colorMode === "cyan" ? "#38d9f9" : "#ffffff"}
        sizeAttenuation
      />
    </points>
  );
}

function CameraFrustumLines({
  poses,
  sceneRadius,
  visible,
}: {
  poses: CameraPose[];
  sceneRadius: number;
  visible: boolean;
}) {
  const geometry = useMemo(() => {
    const vertices: number[] = [];
    const scale = Math.max(sceneRadius * 0.045, 0.02);
    for (const cam of poses) {
      const c2w = poseMatrix(cam.c2w);
      const aspect = cam.width / cam.height;
      const halfH = Math.tan(((cam.fov_y_deg / 2) * Math.PI) / 180) * scale;
      const halfW = halfH * aspect;
      const local = [
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(-halfW, -halfH, -scale),
        new THREE.Vector3(halfW, -halfH, -scale),
        new THREE.Vector3(halfW, halfH, -scale),
        new THREE.Vector3(-halfW, halfH, -scale),
      ].map((v) => v.applyMatrix4(c2w));
      const edges: [number, number][] = [
        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1],
      ];
      for (const [a, b] of edges) {
        vertices.push(local[a].x, local[a].y, local[a].z);
        vertices.push(local[b].x, local[b].y, local[b].z);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    return geo;
  }, [poses, sceneRadius]);
  if (!visible) return null;
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#22d3ee" transparent opacity={0.55} />
    </lineSegments>
  );
}

function CameraMarkers({
  poses,
  sceneRadius,
  visible,
  onPick,
}: {
  poses: CameraPose[];
  sceneRadius: number;
  visible: boolean;
  onPick: (pose: CameraPose) => void;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const size = Math.max(sceneRadius * 0.012, 0.008);
  if (!visible) return null;
  return (
    <group>
      {poses.map((pose) => {
        const c2w = poseMatrix(pose.c2w);
        const pos = new THREE.Vector3().setFromMatrixPosition(c2w);
        return (
          <mesh
            key={pose.id}
            position={pos}
            onPointerDown={(e) => {
              e.stopPropagation();
              onPick(pose);
            }}
            onPointerOver={(e) => {
              e.stopPropagation();
              setHovered(pose.id);
              document.body.style.cursor = "pointer";
            }}
            onPointerOut={() => {
              setHovered(null);
              document.body.style.cursor = "auto";
            }}
          >
            <sphereGeometry args={[hovered === pose.id ? size * 1.8 : size, 14, 14]} />
            <meshBasicMaterial
              color={hovered === pose.id ? "#a5f3fc" : "#22d3ee"}
              transparent
              opacity={0.9}
            />
          </mesh>
        );
      })}
    </group>
  );
}

/** Smoothly flies the camera to a requested pose, then re-enables controls. */
function CameraRig({
  flyTarget,
  controlsRef,
}: {
  flyTarget: FlyTarget | null;
  controlsRef: OrbitControlsRef;
}) {
  const { camera } = useThree();
  const anim = useRef<{
    startPos: THREE.Vector3;
    startQuat: THREE.Quaternion;
    startTarget: THREE.Vector3;
    startTime: number;
  } | null>(null);

  useEffect(() => {
    if (!flyTarget) return;
    anim.current = {
      startPos: camera.position.clone(),
      startQuat: camera.quaternion.clone(),
      startTarget: controlsRef.current?.target.clone() ?? new THREE.Vector3(),
      startTime: performance.now(),
    };
    if (controlsRef.current) controlsRef.current.enabled = false;
  }, [flyTarget, camera, controlsRef]);

  useFrame(() => {
    const a = anim.current;
    if (!a || !flyTarget) return;
    const t = Math.min((performance.now() - a.startTime) / 900, 1);
    const e = smoothstep(t);
    camera.position.lerpVectors(a.startPos, flyTarget.pos, e);
    camera.quaternion.slerpQuaternions(a.startQuat, flyTarget.quat, e);
    controlsRef.current?.target.lerpVectors(a.startTarget, flyTarget.target, e);
    controlsRef.current?.update();
    if (t >= 1) {
      anim.current = null;
      if (controlsRef.current) controlsRef.current.enabled = true;
    }
  });
  return null;
}

function FpsMeter({ onFps }: { onFps: (fps: number) => void }) {
  const acc = useRef({ frames: 0, t0: performance.now() });
  useFrame(() => {
    const now = performance.now();
    acc.current.frames += 1;
    if (now - acc.current.t0 >= 500) {
      onFps(Math.round((acc.current.frames * 1000) / (now - acc.current.t0)));
      acc.current = { frames: 0, t0: now };
    }
  });
  return null;
}

/** Registers the screenshot routine: one fresh render, then PNG download. */
function CaptureBinder({ register }: { register: (fn: () => void) => void }) {
  const { gl, scene, camera } = useThree();
  useEffect(() => {
    register(() => {
      gl.render(scene, camera); // guarantee a fresh frame in the buffer
      gl.domElement.toBlob((blob: Blob | null) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `spatialforge-screenshot-${new Date()
          .toISOString()
          .replace(/[:.]/g, "-")}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      }, "image/png");
    });
  }, [gl, scene, camera, register]);
  return null;
}

function SceneContents({
  ply,
  poses,
  colorMode,
  pointScale,
  showGrid,
  showCameras,
  background,
  autoRotate,
  flyTarget,
  onHoverPoint,
  onFps,
  onPickCamera,
  onHomeReady,
  registerCapture,
}: {
  ply: PlyCloud | null;
  poses: CameraPose[];
  colorMode: ColorMode;
  pointScale: number;
  showGrid: boolean;
  showCameras: boolean;
  background: string;
  autoRotate: boolean;
  flyTarget: FlyTarget | null;
  onHoverPoint: (info: { index: number; xyz: [number, number, number] } | null) => void;
  onFps: (fps: number) => void;
  onPickCamera: (pose: CameraPose) => void;
  onHomeReady: (home: { pos: number[]; quat: number[]; target: number[] }) => void;
  registerCapture: (fn: () => void) => void;
}) {
  const { scene, camera, raycaster } = useThree();
  const controlsRef: OrbitControlsRef = useRef(null);
  const homeSent = useRef(false);

  useEffect(() => {
    scene.background = new THREE.Color(background);
  }, [scene, background]);

  const sceneRadius = useMemo(() => {
    if (!ply) return 1;
    const geo = cloudGeometry(ply);
    geo.computeBoundingSphere();
    const r = geo.boundingSphere?.radius ?? 1;
    geo.dispose();
    return r;
  }, [ply]);

  // frame the cloud once it loads; that framing is also the "home" view
  useEffect(() => {
    if (!ply || !controlsRef.current || homeSent.current) return;
    const geo = cloudGeometry(ply);
    geo.computeBoundingSphere();
    const bs = geo.boundingSphere;
    if (!bs) return;
    const { center, radius } = bs;
    const dir = new THREE.Vector3(0.45, 0.4, 1).normalize();
    const pos = center.clone().add(dir.multiplyScalar(Math.max(radius * 2.2, 0.5)));
    const quat = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().lookAt(pos, center, new THREE.Vector3(0, 1, 0)),
    );
    camera.position.copy(pos);
    camera.quaternion.copy(quat);
    camera.near = Math.max(radius / 500, 0.0005);
    camera.far = radius * 60;
    camera.updateProjectionMatrix();
    controlsRef.current.target.copy(center);
    controlsRef.current.update();
    homeSent.current = true;
    onHomeReady({
      pos: pos.toArray(),
      quat: quat.toArray(),
      target: center.toArray(),
    });
    geo.dispose();
  }, [ply, camera, onHomeReady]);

  // point-picking threshold scaled to the scene
  useEffect(() => {
    (raycaster.params as { Points?: { threshold: number } }).Points = {
      threshold: sceneRadius * 0.008,
    };
  }, [raycaster, sceneRadius]);

  return (
    <>
      <ambientLight intensity={0.85} />
      {showGrid && ply && (
        <gridHelper
          args={[sceneRadius * 4, 28, "#1b2940", "#111827"]}
          position={[0, -(sceneRadius * 0.95), 0]}
        />
      )}
      {ply && (
        <PointCloud
          ply={ply}
          colorMode={colorMode}
          pointScale={pointScale}
          sceneRadius={sceneRadius}
          onHoverPoint={onHoverPoint}
        />
      )}
      <CameraFrustumLines poses={poses} sceneRadius={sceneRadius} visible={showCameras} />
      <CameraMarkers
        poses={poses}
        sceneRadius={sceneRadius}
        visible={showCameras}
        onPick={onPickCamera}
      />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={1.4}
      />
      <CameraRig flyTarget={flyTarget} controlsRef={controlsRef} />
      <FpsMeter onFps={onFps} />
      <CaptureBinder register={registerCapture} />
    </>
  );
}

// ---------------------------------------------------------------------------
// public panel: toolbar + canvas + HUD + toasts
// ---------------------------------------------------------------------------

export interface ViewerPanelProps {
  plyUrl: string | null;
  camerasUrl: string | null;
  fileHint: string;
}

interface Toast {
  id: number;
  msg: string;
}

export default function ViewerPanel({ plyUrl, camerasUrl, fileHint }: ViewerPanelProps) {
  const [ply, setPly] = useState<PlyCloud | null>(null);
  const [poses, setPoses] = useState<CameraPose[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [showCameras, setShowCameras] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [colorMode, setColorMode] = useState<ColorMode>("original");
  const [pointScale, setPointScale] = useState(1.6);
  const [background, setBackground] = useState("#0b0e14");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [flashKey, setFlashKey] = useState(0);
  const [fps, setFps] = useState(0);
  const [hover, setHover] = useState<{ index: number; xyz: [number, number, number] } | null>(
    null,
  );
  const [flyTarget, setFlyTarget] = useState<FlyTarget | null>(null);
  const [lastPoseName, setLastPoseName] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const captureRef = useRef<() => void>(() => {});
  const homeRef = useRef<{ pos: number[]; quat: number[]; target: number[] } | null>(null);

  const toast = useCallback((msg: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-2), { id, msg }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  }, []);

  const flyTo = useCallback(
    (detail: { pos: number[]; quat: number[]; target: number[] }) => {
      setFlyTarget({
        pos: new THREE.Vector3().fromArray(detail.pos),
        quat: new THREE.Quaternion().fromArray(detail.quat),
        target: new THREE.Vector3().fromArray(detail.target),
      });
    },
    [],
  );

  const pickCamera = useCallback(
    (pose: CameraPose) => {
      const c2w = poseMatrix(pose.c2w);
      const pos = new THREE.Vector3().setFromMatrixPosition(c2w);
      const quat = new THREE.Quaternion().setFromRotationMatrix(c2w);
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quat);
      flyTo({
        pos: pos.toArray(),
        quat: quat.toArray(),
        target: pos.clone().add(forward.multiplyScalar(0.5)).toArray(),
      });
      setLastPoseName(pose.name);
    },
    [flyTo],
  );

  const fit = useCallback(() => {
    if (homeRef.current) {
      flyTo(homeRef.current);
      setLastPoseName(null);
    }
  }, [flyTo]);

  useEffect(() => {
    let alive = true;
    setPly(null);
    setError(null);
    if (!plyUrl) return;
    fetch(plyUrl)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => alive && setPly(parseBinaryPly(buf)))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [plyUrl]);

  useEffect(() => {
    let alive = true;
    if (!camerasUrl) {
      setPoses([]);
      return;
    }
    fetch(camerasUrl)
      .then((r) => r.json())
      .then((data: CameraPose[]) => alive && setPoses(data))
      .catch(() => alive && setPoses([]));
    return () => {
      alive = false;
    };
  }, [camerasUrl]);

  const capture = useCallback(() => {
    setFlashKey((k) => k + 1);
    captureRef.current();
    toast("Screenshot saved to your downloads");
  }, [toast]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (
        el &&
        (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT")
      ) {
        return;
      }
      switch (e.key.toLowerCase()) {
        case "g":
          setShowGrid((v) => !v);
          break;
        case "c":
          setShowCameras((v) => !v);
          break;
        case "f":
          fit();
          break;
        case "p":
          capture();
          break;
        case " ":
          e.preventDefault();
          setAutoRotate((v) => !v);
          break;
        case "1":
          setColorMode("original");
          break;
        case "2":
          setColorMode("cyan");
          break;
        case "3":
          setColorMode("depth");
          break;
        default:
          return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [capture, fit]);

  const fmtCoord = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));

  if (error) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-rose-400/80">
        Failed to load scene: {error}
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      {/* toolbar */}
      <div className="absolute top-2.5 left-2.5 right-2.5 z-10 flex items-center gap-1.5 flex-wrap pointer-events-none">
        <div className="pointer-events-auto flex flex-wrap items-center gap-1 max-w-full rounded-lg border border-slate-700/60 bg-[#0b0e14]/85 backdrop-blur px-1.5 py-1 shadow-glow-soft">
          <button
            className={`btn-tool ${showGrid ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Grid (G)"
            onClick={() => setShowGrid((v) => !v)}
          >
            <Grid3x3 className="w-3.5 h-3.5" />
          </button>
          <button
            className={`btn-tool ${showCameras ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Cameras (C) — click a dot to see the scene from that photo"
            onClick={() => setShowCameras((v) => !v)}
          >
            <Crosshair className="w-3.5 h-3.5" />
          </button>
          <button
            className={`btn-tool ${autoRotate ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Turntable (Space)"
            onClick={() => setAutoRotate((v) => !v)}
          >
            {autoRotate ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>
          <button className="btn-tool" title="Fit view (F)" onClick={fit}>
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <div className="w-px h-5 bg-slate-700/70 mx-0.5" />
          <button
            className={`btn-tool !px-1.5 ${colorMode === "original" ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Original photo colors (1)"
            onClick={() => setColorMode("original")}
          >
            Color
          </button>
          <button
            className={`btn-tool !px-1.5 ${colorMode === "cyan" ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Single cyan color (2)"
            onClick={() => setColorMode("cyan")}
          >
            Cyan
          </button>
          <button
            className={`btn-tool !px-1.5 ${colorMode === "depth" ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Color by distance from center (3)"
            onClick={() => setColorMode("depth")}
          >
            Depth
          </button>
          <div className="w-px h-5 bg-slate-700/70 mx-0.5" />
          <input
            type="range"
            min={0.2}
            max={5}
            step={0.1}
            value={pointScale}
            onChange={(e) => setPointScale(parseFloat(e.target.value))}
            className="w-20 accent-cyan-500"
            title="Point size"
          />
          <select
            className="btn-tool !w-auto"
            value={background}
            onChange={(e) => setBackground(e.target.value)}
            title="Background"
          >
            <option value="#0b0e14">Dark</option>
            <option value="#10141d">Navy</option>
            <option value="#111827">Slate</option>
            <option value="#e2e8f0">Light</option>
          </select>
          <button className="btn-tool" title="Screenshot (P)" onClick={capture}>
            <Camera className="w-3.5 h-3.5" />
          </button>
          <button
            className={`btn-tool ${showShortcuts ? "!text-cyan-300 !border-cyan-700/60" : ""}`}
            title="Keyboard shortcuts"
            onClick={() => setShowShortcuts((v) => !v)}
          >
            <Keyboard className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* keyboard cheat sheet */}
      {showShortcuts && (
        <div className="absolute top-14 right-2.5 z-20 hud animate-fadeUp w-60">
          <div className="text-cyan-300/90 mb-1">shortcuts</div>
          <div className="grid grid-cols-[auto_1fr] gap-x-2">
            <kbd className="text-cyan-300">G</kbd>
            <span>grid</span>
            <kbd className="text-cyan-300">C</kbd>
            <span>cameras</span>
            <kbd className="text-cyan-300">F</kbd>
            <span>fit view</span>
            <kbd className="text-cyan-300">P</kbd>
            <span>screenshot</span>
            <kbd className="text-cyan-300">Space</kbd>
            <span>turntable</span>
            <kbd className="text-cyan-300">1/2/3</kbd>
            <span>color modes</span>
            <kbd className="text-slate-500">click</kbd>
            <span>camera dot = view from that photo</span>
          </div>
        </div>
      )}

      {/* canvas */}
      <div className="absolute inset-0 rounded-lg overflow-hidden border border-slate-800/80">
        {flashKey > 0 && (
          <div
            key={flashKey}
            className="absolute inset-0 bg-white animate-flash z-20 pointer-events-none"
          />
        )}
        {!ply && !error && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-500">
            Loading point cloud…
          </div>
        )}
        <Canvas
          camera={{ position: [0, 1.2, 2.4], fov: 55, near: 0.001, far: 200 }}
          gl={{ preserveDrawingBuffer: true, antialias: true }}
        >
          <SceneContents
            ply={ply}
            poses={poses}
            colorMode={colorMode}
            pointScale={pointScale}
            showGrid={showGrid}
            showCameras={showCameras}
            background={background}
            autoRotate={autoRotate}
            flyTarget={flyTarget}
            onHoverPoint={setHover}
            onFps={setFps}
            onPickCamera={pickCamera}
            onHomeReady={(home) => {
              homeRef.current = home;
            }}
            registerCapture={(fn) => {
              captureRef.current = fn;
            }}
          />
        </Canvas>
      </div>

      {/* HUD */}
      <div className="absolute bottom-2.5 left-2.5 z-10 flex items-end gap-2 pointer-events-none">
        <div className="hud pointer-events-auto">
          <span
            className={
              fps >= 50 ? "text-emerald-400" : fps >= 25 ? "text-amber-400" : "text-rose-400"
            }
          >
            {fps} fps
          </span>
          {ply && <span> · {ply.vertexCount.toLocaleString()} pts</span>}
          {poses.length > 0 && <span> · {poses.length} cams</span>}
        </div>
        {hover && (
          <div className="hud animate-fadeUp">
            vertex <span className="text-cyan-300">#{hover.index}</span> ·{" "}
            {fmtCoord(hover.xyz[0])}, {fmtCoord(hover.xyz[1])}, {fmtCoord(hover.xyz[2])}
          </div>
        )}
        {lastPoseName && (
          <div className="hud animate-fadeUp">
            viewing from <span className="text-cyan-300">{lastPoseName}</span>
          </div>
        )}
      </div>

      {/* toasts */}
      <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 z-20 space-y-1.5">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="animate-fadeUp flex items-center gap-2 rounded-md border border-cyan-700/50 bg-[#0b0e14]/90 backdrop-blur px-3 py-1.5 text-xs text-cyan-200"
          >
            <Download className="w-3.5 h-3.5" /> {t.msg}
          </div>
        ))}
      </div>

      {fileHint && (
        <div className="absolute bottom-2.5 right-2.5 z-10 hud pointer-events-none">{fileHint}</div>
      )}
    </div>
  );
}
