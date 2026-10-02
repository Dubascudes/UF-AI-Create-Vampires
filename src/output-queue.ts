export class OutputQueue<T> {
  private readonly pending: T[] = [];
  private running = false;
  private current: Promise<void> = Promise.resolve();

  constructor(
    private readonly worker: (item: T) => Promise<void>,
    private readonly capacity: number,
    private readonly onError?: (error: unknown) => void,
  ) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error('Queue capacity must be positive');
  }

  enqueue(item: T): void {
    this.pending.push(item);
    while (this.pending.length > this.capacity) this.pending.shift();
    if (!this.running) this.current = this.drain();
  }

  async whenIdle(): Promise<void> { await this.current; }
  clear(): void { this.pending.length = 0; }

  private async drain(): Promise<void> {
    this.running = true;
    try {
      while (this.pending.length) {
        const item = this.pending.shift()!;
        try { await this.worker(item); }
        catch (error) { this.onError?.(error); }
      }
    } finally { this.running = false; }
  }
}
