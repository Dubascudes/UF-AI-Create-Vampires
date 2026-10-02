import { describe, expect, it } from 'vitest';
import { decodeFramePacket, encodeFramePacket, isSafeRecordingId, type RecordedFrameMeta } from './xr-recording';

const meta: RecordedFrameMeta = {
  index: 7, timeMs: 1234.5, source: 'rear-xr', cameraToWorld: Array.from({ length: 16 }, (_, i) => i),
  projection: Array(16).fill(0), image: { width: 4, height: 3 },
  depth: { width: 2, height: 1, format: 'luminance-alpha', rawValueToMeters: 0.001, normDepthBufferFromNormView: Array(16).fill(1) },
};

describe('frame packets', () => {
  it('round-trips metadata, image, and depth', () => {
    const packet = encodeFramePacket(meta, new Uint8Array([1, 2, 3]), new Uint8Array([9, 8, 7, 6]));
    const decoded = decodeFramePacket(packet);
    expect(decoded.meta).toEqual(meta);
    expect(Array.from(decoded.image)).toEqual([1, 2, 3]);
    expect(Array.from(decoded.depth!)).toEqual([9, 8, 7, 6]);
  });

  it('allows front-camera frames without pose or depth', () => {
    const front: RecordedFrameMeta = { ...meta, source: 'front-camera', cameraToWorld: null, projection: null, depth: null,
      orientation: { kind: 'quaternion', value: [0, 0, 0, 1], screenAngle: 0 } };
    const decoded = decodeFramePacket(encodeFramePacket(front, new Uint8Array([5])));
    expect(decoded.meta).toEqual(front);
    expect(decoded.depth).toBeNull();
  });

  it('rejects truncated or foreign packets', () => {
    const packet = encodeFramePacket(meta, new Uint8Array([1, 2, 3]));
    expect(() => decodeFramePacket(packet.subarray(0, packet.length - 1))).toThrow();
    expect(() => decodeFramePacket(new Uint8Array(16))).toThrow();
  });
});

describe('recording ids', () => {
  it('rejects path traversal', () => {
    expect(isSafeRecordingId('2026-09-25T18-00-00')).toBe(true);
    expect(isSafeRecordingId('../etc')).toBe(false);
    expect(isSafeRecordingId('')).toBe(false);
  });
});
