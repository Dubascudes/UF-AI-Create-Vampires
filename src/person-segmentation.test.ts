import { describe, expect, it } from 'vitest';
import { decodeYoloMaskResponse, decodeYoloStreamResponse, encodeYoloStreamFrame } from './person-segmentation';

describe('local YOLO person mask', () => {
  it('round-trips a raw browser frame and binary mask without JPEG encoding', () => {
    const frame = { width: 2, height: 1, data: Uint8ClampedArray.from([10, 20, 30, 255, 40, 50, 60, 255]) } as ImageData;
    const request = encodeYoloStreamFrame(7, frame);
    expect(new DataView(request).getUint32(0, true)).toBe(7);
    expect(Array.from(new Uint8Array(request, 8))).toEqual(Array.from(frame.data));
    const response = new ArrayBuffer(14);
    const header = new DataView(response);
    header.setUint32(0, 7, true);
    header.setUint16(4, 2, true);
    header.setUint16(6, 1, true);
    header.setFloat32(8, 4.25, true);
    new Uint8Array(response, 12).set([255, 0]);
    const decoded = decodeYoloStreamResponse(response);
    expect(decoded.requestId).toBe(7);
    expect(Array.from(decoded.values)).toEqual([1, 0]);
    expect(decoded.inferenceMs).toBe(4.25);
  });

  it('decodes the binary person mask with its source dimensions and timing', async () => {
    const response = new Response(Uint8Array.from([0, 255, 255, 0]), {
      headers: { 'X-Mask-Width': '2', 'X-Mask-Height': '2', 'X-Inference-Ms': '14.6' },
    });
    const result = await decodeYoloMaskResponse(response);
    expect(Array.from(result.values)).toEqual([0, 1, 1, 0]);
    expect([result.width, result.height, result.inferenceMs]).toEqual([2, 2, 14.6]);
  });

  it('rejects a truncated mask rather than treating it as an empty room', async () => {
    const response = new Response(Uint8Array.from([255]), {
      headers: { 'X-Mask-Width': '2', 'X-Mask-Height': '2' },
    });
    await expect(decodeYoloMaskResponse(response)).rejects.toThrow('dimensions');
  });
});
