/**
 * CSR (Compressed Sparse Row) implementation of ConnectivityStore.
 *
 * CSR is the default backing store for V1. It is memory-efficient for sparse
 * connectivity and fast for iterating outgoing synapses (the primary access pattern
 * during spike propagation).
 *
 * V3 will introduce DynamicConnectivityStore (COO-backed) for structural plasticity.
 * The abstraction boundary (ConnectivityStore interface) ensures this transition
 * does not require changes outside packages/network.
 *
 * Memory layout (N sources, M targets, S synapses):
 *   row_ptr:  [N+1] Int32Array   — row_ptr[i] = start index of source i's synapses
 *   col_idx:  [S]   Int32Array   — target neuron index
 *   weights:  [S]   Float32Array — synaptic weight
 *   delays:   [S]   Float32Array — propagation delay in ms
 *   receptor: [S]   Uint8Array   — receptor type enum index
 *
 * Post-trace is indexed by TARGET neuron (not by synapse) — one trace per target.
 */

import type {
  ConnectivityStore,
  SynapseView,
  ReceptorType,
  ConnectivityRule,
  WeightDistribution,
  DelayDistribution,
} from '@snn/types';
import { SeededRNG, sampleWeights, sampleDelays } from '@snn/config';

const RECEPTOR_IDS: Record<ReceptorType, number> = {
  AMPA: 0,
  GABA_A: 1,
  NMDA: 2,
  GABA_B: 3,
};

const RECEPTOR_FROM_ID: ReceptorType[] = ['AMPA', 'GABA_A', 'NMDA', 'GABA_B'];

export class CSRConnectivityStore implements ConnectivityStore {
  readonly sourceSize: number;
  readonly targetSize: number;

  private readonly rowPtr: Int32Array;
  private readonly colIdx: Int32Array;
  private readonly _weights: Float32Array;
  private readonly _delays: Float32Array;
  private readonly _receptor: Uint8Array;

  get synapseCount(): number {
    return this.colIdx.length;
  }

  constructor(
    sourceSize: number,
    targetSize: number,
    rowPtr: Int32Array,
    colIdx: Int32Array,
    weights: Float32Array,
    delays: Float32Array,
    receptor: Uint8Array
  ) {
    this.sourceSize = sourceSize;
    this.targetSize = targetSize;
    this.rowPtr = rowPtr;
    this.colIdx = colIdx;
    this._weights = weights;
    this._delays = delays;
    this._receptor = receptor;
  }

  getOutgoing(sourceIndex: number): SynapseView[] {
    const start = this.rowPtr[sourceIndex] as number;
    const end = this.rowPtr[sourceIndex + 1] as number;
    const results: SynapseView[] = [];

    for (let idx = start; idx < end; idx++) {
      results.push(this.makeSynapseView(idx, sourceIndex));
    }
    return results;
  }

  getIncoming(targetIndex: number): SynapseView[] {
    const results: SynapseView[] = [];
    // CSR is not efficient for incoming queries — linear scan
    // For V3 plasticity requiring efficient incoming access, use DynamicConnectivityStore
    for (let src = 0; src < this.sourceSize; src++) {
      const start = this.rowPtr[src] as number;
      const end = this.rowPtr[src + 1] as number;
      for (let idx = start; idx < end; idx++) {
        if (this.colIdx[idx] === targetIndex) {
          results.push(this.makeSynapseView(idx, src));
        }
      }
    }
    return results;
  }

  forEach(callback: (synapse: SynapseView) => void): void {
    for (let src = 0; src < this.sourceSize; src++) {
      const start = this.rowPtr[src] as number;
      const end = this.rowPtr[src + 1] as number;
      for (let idx = start; idx < end; idx++) {
        callback(this.makeSynapseView(idx, src));
      }
    }
  }

  getWeight(synapseIndex: number): number {
    return this._weights[synapseIndex] as number;
  }

  setWeight(synapseIndex: number, weight: number): void {
    this._weights[synapseIndex] = weight;
  }

  clampWeights(minWeight: number, maxWeight: number): void {
    for (let i = 0; i < this._weights.length; i++) {
      this._weights[i] = Math.max(minWeight, Math.min(maxWeight, this._weights[i] as number));
    }
  }

  getWeightsSnapshot(): Float32Array {
    return this._weights.slice();
  }

  getDelay(synapseIndex: number): number {
    return this._delays[synapseIndex] as number;
  }

  getReceptorType(synapseIndex: number): ReceptorType {
    return RECEPTOR_FROM_ID[this._receptor[synapseIndex] as number] ?? 'AMPA';
  }

  private makeSynapseView(idx: number, sourceIndex: number): SynapseView {
    const store = this;
    return {
      index: idx,
      sourceIndex,
      targetIndex: store.colIdx[idx] as number,
      delay: store._delays[idx] as number,
      receptorType: RECEPTOR_FROM_ID[store._receptor[idx] as number] ?? 'AMPA',
      get weight() { return store._weights[idx] as number; },
      set weight(v: number) { store._weights[idx] = v; },
    };
  }
}

/** Build a CSRConnectivityStore from a connectivity rule */
export function buildCSRStore(
  sourceSize: number,
  targetSize: number,
  rule: ConnectivityRule,
  weightDist: WeightDistribution,
  delayDist: DelayDistribution,
  receptorType: ReceptorType,
  rng: SeededRNG
): CSRConnectivityStore {
  // Build COO first, then convert to CSR
  const srcs: number[] = [];
  const tgts: number[] = [];

  switch (rule.type) {
    case 'all_to_all':
      for (let s = 0; s < sourceSize; s++) {
        for (let t = 0; t < targetSize; t++) {
          srcs.push(s);
          tgts.push(t);
        }
      }
      break;

    case 'one_to_one': {
      const n = Math.min(sourceSize, targetSize);
      for (let i = 0; i < n; i++) {
        srcs.push(i);
        tgts.push(i);
      }
      break;
    }

    case 'random': {
      const localRng = rule.seed !== undefined ? new SeededRNG(rule.seed) : rng;
      for (let s = 0; s < sourceSize; s++) {
        for (let t = 0; t < targetSize; t++) {
          if (localRng.random() < rule.probability) {
            srcs.push(s);
            tgts.push(t);
          }
        }
      }
      break;
    }

    case 'fixed_number_pre': {
      const localRng = rule.seed !== undefined ? new SeededRNG(rule.seed) : rng;
      const allTargets = Array.from({ length: targetSize }, (_, i) => i);
      for (let s = 0; s < sourceSize; s++) {
        const n = Math.min(rule.n, targetSize);
        const chosen = localRng.shuffle([...allTargets]).slice(0, n);
        for (const t of chosen) {
          srcs.push(s);
          tgts.push(t);
        }
      }
      break;
    }

    case 'custom':
      for (const [s, t] of rule.connections) {
        srcs.push(s);
        tgts.push(t);
      }
      break;

    case 'topographic':
      // Distance-dependent Gaussian connectivity
      for (let s = 0; s < sourceSize; s++) {
        for (let t = 0; t < targetSize; t++) {
          const dist = Math.abs(s / sourceSize - t / targetSize);
          const prob = Math.exp(-(dist * dist) / (2 * rule.sigma * rule.sigma));
          if (rng.random() < prob) {
            srcs.push(s);
            tgts.push(t);
          }
        }
      }
      break;
  }

  const S = srcs.length;
  const weights = sampleWeights(weightDist, S, rng);
  const delays = sampleDelays(delayDist, S, rng);
  const receptor = new Uint8Array(S).fill(RECEPTOR_IDS[receptorType]);

  // Sort by source index (required for CSR)
  const order = Array.from({ length: S }, (_, i) => i)
    .sort((a, b) => (srcs[a] as number) - (srcs[b] as number));

  const sortedSrcs = order.map(i => srcs[i] as number);
  const sortedTgts = new Int32Array(order.map(i => tgts[i] as number));
  const sortedWeights = new Float32Array(order.map(i => weights[i] as number));
  const sortedDelays = new Float32Array(order.map(i => delays[i] as number));
  const sortedReceptor = new Uint8Array(order.map(i => receptor[i] as number));

  // Build row_ptr
  const rowPtr = new Int32Array(sourceSize + 1);
  for (const src of sortedSrcs) {
    rowPtr[src + 1]++;
  }
  for (let i = 1; i <= sourceSize; i++) {
    rowPtr[i] += rowPtr[i - 1] as number;
  }

  return new CSRConnectivityStore(
    sourceSize,
    targetSize,
    rowPtr,
    sortedTgts,
    sortedWeights,
    sortedDelays,
    sortedReceptor
  );
}
