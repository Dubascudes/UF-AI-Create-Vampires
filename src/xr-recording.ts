/**
 * One recorded AR frame travels from the phone to the dev server as a single binary packet:
 * "ABXR", then little-endian u32 lengths of the JSON metadata, JPEG image and raw depth buffer,
 * followed by those three byte ranges.
 */
const MAGIC = 0x52584241; // "ABXR" read as little-endian u32
const HEADER_BYTES = 16;

export interface RecordedFrameMeta {
  index: number;
  timeMs: number;
  /** rear-xr: world-facing camera in a WebXR session. front-camera: selfie camera via getUserMedia, no tracking or depth. */
  source: 'rear-xr' | 'front-camera';
  /** Camera-to-world (local reference space), column-major 4×4; null without WebXR tracking. */
  cameraToWorld: number[] | null;
  /** View projection matrix, column-major 4×4; null without WebXR. */
  projection: number[] | null;
  /**
   * Device rotation from the phone's motion sensors (front camera only): a RelativeOrientationSensor
   * quaternion [x, y, z, w], or deviceorientation [alpha, beta, gamma] degrees; screenAngle is the screen rotation.
   */
  orientation?: { kind: 'quaternion' | 'euler-deg'; value: number[]; screenAngle: number } | null;
  image: { width: number; height: number };
  depth: null | {
    width: number;
    height: number;
    format: string;
    rawValueToMeters: number;
    /** Maps normalized view coordinates (origin top-left, y down) to normalized depth coordinates. */
    normDepthBufferFromNormView: number[];
  };
}

export interface FramePacket { meta: RecordedFrameMeta; image: Uint8Array; depth: Uint8Array | null }

export function encodeFramePacket(meta: RecordedFrameMeta, image: Uint8Array, depth?: Uint8Array): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(meta));
  const depthLength = depth?.length ?? 0;
  const packet = new Uint8Array(HEADER_BYTES + json.length + image.length + depthLength);
  const view = new DataView(packet.buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, json.length, true);
  view.setUint32(8, image.length, true);
  view.setUint32(12, depthLength, true);
  packet.set(json, HEADER_BYTES);
  packet.set(image, HEADER_BYTES + json.length);
  if (depth) packet.set(depth, HEADER_BYTES + json.length + image.length);
  return packet;
}

export function decodeFramePacket(packet: Uint8Array): FramePacket {
  if (packet.length < HEADER_BYTES) throw new Error('Frame packet header is missing');
  const view = new DataView(packet.buffer, packet.byteOffset, packet.byteLength);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('Frame packet has the wrong magic number');
  const jsonLength = view.getUint32(4, true);
  const imageLength = view.getUint32(8, true);
  const depthLength = view.getUint32(12, true);
  if (HEADER_BYTES + jsonLength + imageLength + depthLength !== packet.length) {
    throw new Error('Frame packet lengths do not match its size');
  }
  const imageStart = HEADER_BYTES + jsonLength;
  const depthStart = imageStart + imageLength;
  const meta = JSON.parse(new TextDecoder().decode(packet.subarray(HEADER_BYTES, imageStart))) as RecordedFrameMeta;
  if (!Number.isInteger(meta.index) || meta.index < 0) throw new Error('Frame packet has an invalid index');
  return {
    meta,
    image: packet.subarray(imageStart, depthStart),
    depth: depthLength ? packet.subarray(depthStart) : null,
  };
}

export function isSafeRecordingId(id: string): boolean {
  return /^[A-Za-z0-9-]{1,64}$/.test(id);
}
