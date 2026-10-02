import { describe, expect, it } from 'vitest';
import { liveOpacity, personCoverage } from './mask';

describe('person mask', () => {
  it('keeps background and removes a confidently detected person', () => {
    expect(liveOpacity(0, 0.5)).toBe(255);
    expect(liveOpacity(1, 0.5)).toBe(0);
    expect(liveOpacity(0.5, 0.5)).toBeGreaterThan(0);
    expect(liveOpacity(0.5, 0.5)).toBeLessThan(255);
  });

  it('measures occupancy before capturing an empty scene', () => {
    expect(personCoverage(new Float32Array([0.1, 0.7, 0.9, 0.2]))).toBe(0.5);
    expect(personCoverage(new Float32Array())).toBe(0);
  });
});
