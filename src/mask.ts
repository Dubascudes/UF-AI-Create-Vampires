/** Convert person confidence into a soft opacity for the live camera image. */
export function liveOpacity(confidence: number, threshold: number): number {
  const lower = Math.max(0, threshold - 0.16);
  const upper = Math.min(1, threshold + 0.16);
  const t = Math.max(0, Math.min(1, (confidence - lower) / (upper - lower)));
  const smooth = t * t * (3 - 2 * t);
  return Math.round((1 - smooth) * 255);
}

export function personCoverage(confidences: Float32Array, threshold = 0.5): number {
  if (confidences.length === 0) return 0;
  let count = 0;
  for (const confidence of confidences) if (confidence >= threshold) count++;
  return count / confidences.length;
}
