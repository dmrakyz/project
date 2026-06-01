/**
 * Circular buffer implementing axonal conduction delays.
 *
 * All spike delivery goes through this delay line. Zero-delay is architecturally
 * forbidden — minimum delay is one simulation timestep (dt).
 *
 * Implementation: one circular buffer per projection.
 *   - Slot count = ceil(maxDelay / dt) + 1
 *   - At timestep t, slot (t mod slotCount) holds spikes due for delivery at time t
 *   - A spike with delay d is written to slot ((writeHead + delaySteps) mod slotCount)
 *   - advance() moves writeHead forward; dequeue() returns and clears the current slot
 *
 * Memory: O(maxDelay/dt × averageSpikesPerStep) per projection.
 * For maxDelay=20ms, dt=0.1ms: 200 slots.
 */

import type { DelayLine } from '@snn/types';

export class CircularDelayLine implements DelayLine {
  readonly maxDelaySteps: number;
  writeHead: number = 0;
  readonly buffer: Array<number[]>;

  constructor(maxDelayMs: number, dt: number) {
    // +1 to avoid wrap-around collision between write and read heads
    this.maxDelaySteps = Math.ceil(maxDelayMs / dt) + 1;
    this.buffer = Array.from({ length: this.maxDelaySteps }, () => []);
  }

  enqueue(neuronIndex: number, delaySteps: number): void {
    const slot = (this.writeHead + delaySteps) % this.maxDelaySteps;
    (this.buffer[slot] as number[]).push(neuronIndex);
  }

  dequeue(): readonly number[] {
    const current = this.buffer[this.writeHead] as number[];
    return current;
  }

  advance(): void {
    // Clear the slot we just delivered from (it becomes the write target for future spikes)
    (this.buffer[this.writeHead] as number[]).length = 0;
    this.writeHead = (this.writeHead + 1) % this.maxDelaySteps;
  }

  /** How many spikes are queued across all slots (for diagnostic purposes) */
  totalQueued(): number {
    return this.buffer.reduce((sum, slot) => sum + slot.length, 0);
  }
}
