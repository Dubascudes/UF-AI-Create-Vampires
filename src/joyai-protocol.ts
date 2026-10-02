export interface OutputMetadata {
  source_seq?: number;
  t_capture_ms?: number;
  profile?: { passthrough?: boolean };
}

export function createStartMessage(prompt: string, width: number, height: number, fps: number): Record<string, unknown> {
  return {
    type: 'start',
    prompt,
    width,
    height,
    fps,
    num_inference_steps: 2,
    use_pe: false,
    gate_enabled: false,
    freeze_kv_on_static: false,
    kv_reset_frames: 0,
    input_codec: 'mjpeg',
    output_codec: 'mjpeg',
    output_quality: 70,
    seed: 42,
  };
}

export function shouldSendInputFrame(sent: number, received: number, lastSentAt: number, now: number): boolean {
  return sent - received < 64 || now - lastSentAt >= 1000;
}

export function shouldDisplayOutput(meta: OutputMetadata | null, lastCaptureMs: number): boolean {
  return !!meta && !meta.profile?.passthrough && Number.isFinite(meta.t_capture_ms)
    && Number(meta.t_capture_ms) > lastCaptureMs;
}
