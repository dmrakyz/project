/**
 * Weight distribution monitor — tracks synaptic weight statistics over time.
 *
 * Critical research tool: weight entropy over time reveals whether homeostasis
 * is erasing STDP-learned structure. If weight entropy remains constant despite
 * experience, homeostasis timescale is too aggressive.
 *
 * Snapshots are taken at configurable intervals — not every timestep.
 * This keeps memory usage bounded.
 */

import type { Projection, WeightSnapshot } from '@snn/types';

export interface WeightMonitorConfig {
  /** Snapshot interval in simulation steps */
  readonly intervalSteps: number;
  /** Maximum snapshots to retain per projection */
  readonly maxSnapshots?: number;
}

export class WeightMonitor {
  private readonly intervalSteps: number;
  private readonly maxSnapshots: number;
  private stepCounter = 0;
  private readonly snapshots: Map<number, WeightSnapshot[]> = new Map();

  constructor(config: WeightMonitorConfig = { intervalSteps: 1000 }) {
    this.intervalSteps = config.intervalSteps;
    this.maxSnapshots = config.maxSnapshots ?? 1000;
  }

  onTimestep(projections: ReadonlyMap<number, Projection>, t: number): void {
    this.stepCounter++;
    if (this.stepCounter < this.intervalSteps) return;
    this.stepCounter = 0;

    for (const [projId, projection] of projections) {
      const weights = projection.store.getWeightsSnapshot();
      if (weights.length === 0) continue;

      const stats = computeWeightStats(weights);
      const snapshot: WeightSnapshot = {
        t,
        projectionId: projId,
        weights: weights.slice(),
        ...stats,
      };

      if (!this.snapshots.has(projId)) {
        this.snapshots.set(projId, []);
      }
      const list = this.snapshots.get(projId)!;
      list.push(snapshot);
      if (list.length > this.maxSnapshots) list.shift();
    }
  }

  getSnapshots(projectionId: number): WeightSnapshot[] {
    return this.snapshots.get(projectionId) ?? [];
  }

  /** Compute weight entropy for a projection at the latest snapshot */
  getWeightEntropy(projectionId: number): number {
    const snaps = this.snapshots.get(projectionId);
    if (!snaps || snaps.length === 0) return 0;
    const latest = snaps[snaps.length - 1]!;
    return computeEntropy(latest.weights);
  }

  getAllLatestSnapshots(): Map<number, WeightSnapshot> {
    const result = new Map<number, WeightSnapshot>();
    for (const [projId, snaps] of this.snapshots) {
      if (snaps.length > 0) result.set(projId, snaps[snaps.length - 1]!);
    }
    return result;
  }
}

function computeWeightStats(weights: Float32Array): { mean: number; std: number; min: number; max: number } {
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < weights.length; i++) {
    const w = weights[i] as number;
    sum += w;
    if (w < min) min = w;
    if (w > max) max = w;
  }

  const mean = sum / weights.length;
  let variance = 0;
  for (let i = 0; i < weights.length; i++) {
    const d = (weights[i] as number) - mean;
    variance += d * d;
  }
  const std = Math.sqrt(variance / weights.length);

  return { mean, std, min, max };
}

/** Shannon entropy of weight distribution (discretized into 20 bins) */
function computeEntropy(weights: Float32Array, bins: number = 20): number {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i] as number;
    if (w < min) min = w;
    if (w > max) max = w;
  }

  if (max === min) return 0;
  const binSize = (max - min) / bins;
  const counts = new Float64Array(bins);

  for (let i = 0; i < weights.length; i++) {
    const bin = Math.min(bins - 1, Math.floor(((weights[i] as number) - min) / binSize));
    counts[bin]++;
  }

  let entropy = 0;
  const n = weights.length;
  for (let b = 0; b < bins; b++) {
    const p = (counts[b] as number) / n;
    if (p > 0) entropy -= p * Math.log2(p);
  }
  return entropy;
}
