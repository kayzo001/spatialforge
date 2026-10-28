/** Camera pose as stored in cameras.json (OpenGL camera-to-world convention). */
export interface CameraPose {
  id: number;
  name: string;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  c2w: number[][];
  fov_y_deg: number;
}
