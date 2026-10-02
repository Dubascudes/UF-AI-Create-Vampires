import { describe, expect, it } from 'vitest';
import { createStartMessage, shouldDisplayOutput, shouldSendInputFrame } from './joyai-protocol';

describe('generated-frame streaming protocol', () => {
  it('starts an unconditional editing session with no person gate or raw-frame fallback', () => {
    const start = createStartMessage('Remove every person. Keep the room and objects.', 1248, 720, 30);
    expect(start).toMatchObject({
      type: 'start',
      gate_enabled: false,
      use_pe: false,
      freeze_kv_on_static: false,
      kv_reset_frames: 0,
      input_codec: 'mjpeg',
      output_codec: 'mjpeg',
      width: 1248,
      height: 720,
      fps: 30,
    });
  });

  it('ignores missing or stale output metadata and accepts newer generated frames', () => {
    expect(shouldDisplayOutput(null, 0)).toBe(false);
    expect(shouldDisplayOutput({ t_capture_ms: 100 }, 100)).toBe(false);
    expect(shouldDisplayOutput({ t_capture_ms: 101, profile: { passthrough: true } }, 100)).toBe(false);
    expect(shouldDisplayOutput({ t_capture_ms: 101 }, 100)).toBe(true);
  });

  it('paces input when the model falls behind while retaining a keepalive frame', () => {
    expect(shouldSendInputFrame(63, 0, 1000, 1100)).toBe(true);
    expect(shouldSendInputFrame(64, 0, 1000, 1100)).toBe(false);
    expect(shouldSendInputFrame(64, 0, 1000, 2000)).toBe(true);
    expect(shouldSendInputFrame(100, 50, 1000, 1100)).toBe(true);
  });
});
