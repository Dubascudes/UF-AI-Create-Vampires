import { describe, expect, it } from 'vitest';
import { OutputQueue } from './output-queue';

describe('generated output queue', () => {
  it('processes a burst sequentially instead of discarding all but the first frame', async () => {
    const seen: number[] = [];
    const queue = new OutputQueue<number>(async frame => { seen.push(frame); }, 8);
    for (let frame = 1; frame <= 8; frame++) queue.enqueue(frame);
    await queue.whenIdle();
    expect(seen).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('drops old queued frames when processing falls behind and preserves the latest frame', async () => {
    let release!: () => void;
    const firstBlocked = new Promise<void>(resolve => { release = resolve; });
    const seen: number[] = [];
    const queue = new OutputQueue<number>(async frame => {
      if (frame === 1) await firstBlocked;
      seen.push(frame);
    }, 2);
    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);
    queue.enqueue(4);
    release();
    await queue.whenIdle();
    expect(seen).toEqual([1, 3, 4]);
  });
});
