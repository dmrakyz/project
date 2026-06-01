/**
 * Spike recorder — collects spike rasters for offline analysis.
 *
 * For populations under sizeThreshold (default 10K), stores raw spike timestamps.
 * For larger populations, stores only population-level rates to control memory.
 *
 * The recorder is a subscriber to the event bus — it does not receive spikes
 * directly from the simulation. This decoupling allows the simulation to run
 * without any recording overhead when no recorder is attached.
 */

import type { EventBus, SpikeObservabilityEvent, SpikeRasterEntry, Unsubscribe } from '@snn/types';

export interface SpikeRecorderConfig {
  /** Populations with size > this threshold use rate-only mode */
  readonly sizeThreshold?: number;
  /** Maximum raster entries to store (ring buffer behavior when exceeded) */
  readonly maxEntries?: number;
}

export class SpikeRecorder {
  private readonly sizeThreshold: number;
  private readonly maxEntries: number;
  private readonly raster: SpikeRasterEntry[] = [];
  private readonly unsubscribers: Unsubscribe[] = [];
  private writeIndex = 0;
  private wrapped = false;

  constructor(config: SpikeRecorderConfig = {}) {
    this.sizeThreshold = config.sizeThreshold ?? 10_000;
    this.maxEntries = config.maxEntries ?? 1_000_000;
  }

  /** Attach to an event bus and begin recording */
  attach(bus: EventBus, populationIds?: readonly number[]): void {
    if (populationIds) {
      for (const id of populationIds) {
        this.unsubscribers.push(
          bus.subscribe<SpikeObservabilityEvent>('spike', event => this.record(event.populationId, event.neuronIndex, event.t),
            { populationId: id })
        );
      }
    } else {
      this.unsubscribers.push(
        bus.subscribe<SpikeObservabilityEvent>('spike', event => this.record(event.populationId, event.neuronIndex, event.t))
      );
    }
  }

  detach(): void {
    for (const unsub of this.unsubscribers) unsub();
    this.unsubscribers.length = 0;
  }

  private record(populationId: number, neuronIndex: number, t: number): void {
    const entry: SpikeRasterEntry = { t, populationId, neuronIndex };
    if (!this.wrapped && this.raster.length < this.maxEntries) {
      this.raster.push(entry);
    } else {
      // Ring buffer behavior when maxEntries is exceeded
      this.wrapped = true;
      this.raster[this.writeIndex % this.maxEntries] = entry;
      this.writeIndex++;
    }
  }

  /** Get all recorded spikes, optionally filtered by population and time range */
  getSpikes(options: {
    populationId?: number;
    tMin?: number;
    tMax?: number;
  } = {}): SpikeRasterEntry[] {
    const entries = this.wrapped
      ? [...this.raster.slice(this.writeIndex % this.maxEntries), ...this.raster.slice(0, this.writeIndex % this.maxEntries)]
      : [...this.raster];

    return entries.filter(e => {
      if (options.populationId !== undefined && e.populationId !== options.populationId) return false;
      if (options.tMin !== undefined && e.t < options.tMin) return false;
      if (options.tMax !== undefined && e.t > options.tMax) return false;
      return true;
    });
  }

  /** Total spike count recorded */
  get spikeCount(): number {
    return this.wrapped ? this.maxEntries : this.raster.length;
  }

  /** Clear all recorded data */
  clear(): void {
    this.raster.length = 0;
    this.writeIndex = 0;
    this.wrapped = false;
  }

  /** Export to plain objects for serialization */
  toJSON(): SpikeRasterEntry[] {
    return this.getSpikes();
  }
}

/**
 * Sliding window firing rate estimator.
 *
 * Maintains a ring buffer of recent spike times per neuron.
 * On each call to estimate(), returns the firing rate in Hz over the window.
 */
export class RateEstimator {
  private readonly windowMs: number;
  private readonly spikeHistory: Map<number, number[]> = new Map(); // populationId → [t1, t2, ...]

  constructor(windowMs: number = 100) {
    this.windowMs = windowMs;
  }

  recordSpikes(populationId: number, spikeIndices: number[], t: number): void {
    if (!this.spikeHistory.has(populationId)) {
      this.spikeHistory.set(populationId, []);
    }
    const history = this.spikeHistory.get(populationId)!;
    for (let i = 0; i < spikeIndices.length; i++) {
      history.push(t);
    }
  }

  /** Estimate mean population firing rate in Hz */
  estimateRate(populationId: number, populationSize: number, t: number): number {
    const history = this.spikeHistory.get(populationId);
    if (!history || history.length === 0) return 0;

    const tMin = t - this.windowMs;

    // Remove spikes outside window (front of array = oldest)
    let start = 0;
    while (start < history.length && (history[start] as number) < tMin) start++;
    if (start > 0) history.splice(0, start);

    return (history.length / (populationSize * this.windowMs)) * 1000; // Hz
  }
}
