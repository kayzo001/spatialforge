import * as THREE from "three";

export interface PlyCloud {
  positions: Float32Array;
  colors: Float32Array;
  vertexCount: number;
}

const PLY_HEADER_END = "end_header";

/**
 * Minimal binary_little_endian PLY reader for point clouds with x/y/z and
 * red/green/blue (plus optional float properties such as `error`). Gaussian
 * splat PLYs are handled by the dedicated splat renderer instead.
 */
export function parseBinaryPly(buffer: ArrayBuffer): PlyCloud {
  const bytes = new Uint8Array(buffer);
  const headerText = new TextDecoder("ascii").decode(
    bytes.subarray(0, Math.min(bytes.length, 64 * 1024)),
  );
  const endIdx = headerText.indexOf(PLY_HEADER_END);
  if (endIdx < 0) throw new Error("PLY: end_header not found");
  const header = headerText.slice(0, endIdx);
  const dataStart = endIdx + PLY_HEADER_END.length + 1; // + newline

  const lines = header.split("\n").map((l) => l.trim());
  if (!lines.some((l) => l.startsWith("format binary_little_endian"))) {
    throw new Error("PLY: only binary_little_endian is supported");
  }

  let vertexCount = 0;
  const properties: { name: string; type: string }[] = [];
  let inVertex = false;
  for (const line of lines) {
    const tokens = line.split(/\s+/);
    if (tokens[0] === "element") {
      inVertex = tokens[1] === "vertex";
      if (inVertex) vertexCount = parseInt(tokens[2], 10);
    } else if (tokens[0] === "property" && inVertex) {
      if (tokens[1] === "list") continue;
      properties.push({ name: tokens[2], type: tokens[1] });
    }
  }

  const typeSizes: Record<string, number> = {
    float: 4, float32: 4, double: 8, float64: 8,
    uchar: 1, uint8: 1, char: 1, int8: 1,
    ushort: 2, uint16: 2, short: 2, int16: 2,
    int: 4, int32: 4, uint: 4, uint32: 4,
  };
  const stride = properties.reduce((s, p) => s + (typeSizes[p.type] ?? 4), 0);
  if (vertexCount * stride + dataStart > bytes.length + stride) {
    // allow trailing slack but fail on clearly truncated files
    if (vertexCount * stride > bytes.length - dataStart) {
      throw new Error("PLY: truncated vertex data");
    }
  }

  const want = new Set(["x", "y", "z", "red", "green", "blue"]);
  const positions = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);

  const view = new DataView(buffer, dataStart);
  let offset = 0;
  for (let i = 0; i < vertexCount; i++) {
    let px = 0, py = 0, pz = 0, pr = 255, pg = 255, pb = 255;
    for (const prop of properties) {
      const size = typeSizes[prop.type];
      switch (prop.name) {
        case "x": px = view.getFloat32(offset, true); break;
        case "y": py = view.getFloat32(offset, true); break;
        case "z": pz = view.getFloat32(offset, true); break;
        case "red": pr = view.getUint8(offset); break;
        case "green": pg = view.getUint8(offset); break;
        case "blue": pb = view.getUint8(offset); break;
        default: break; // skip unsupported scalars
      }
      offset += size;
    }
    positions[i * 3] = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;
    colors[i * 3] = pr / 255;
    colors[i * 3 + 1] = pg / 255;
    colors[i * 3 + 2] = pb / 255;
  }
  void want;
  return { positions, colors, vertexCount };
}

export function cloudGeometry(ply: PlyCloud): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(ply.positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(ply.colors, 3));
  geometry.computeBoundingSphere();
  return geometry;
}
