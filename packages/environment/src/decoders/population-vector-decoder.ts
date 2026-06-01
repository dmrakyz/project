/**
 * Population vector decoder — converts spike patterns to motor commands.
 *
 * Uses population vector decoding (Georgopoulos et al., 1986): each motor neuron
 * has a "preferred direction" or "preferred value." The output command is the
 * weighted sum of preferred directions, weighted by recent firing rates.
 *
 * This is biologically plausible and avoids the ANN anti-pattern of adding
 * a linear readout layer (W @ activation_vector). The decoding is fully determined
 * by the preferred direction assignment — the weights driving decoding are the
 * actual synaptic weights causing spikes, not a separate learned matrix.
 *
 * The 20–50ms integration window is a deliberate biological constraint:
 * motor commands execute 20–50ms after the "decision." This delay is a feature,
 * not a bug — it matches biological sensorimotor latencies.
 */

import type { SpikeDecoder } from '@snn/types';

export interface PopulationVectorConfig {
  /** Number of neurons in the motor population */
  readonly n: number;
  /** Preferred values/directions for each neuron */
  readonly preferredValues: Float32Array;
  /** Sliding window size in ms for rate estimation */
  readonly windowMs?: number;
  /** Whether to normalize output to [minOutput, maxOutput] */
  readonly minOutput?: number;
  readonly maxOutput?: number;
}

export class PopulationVectorDecoder implements SpikeDecoder {
  readonly name = 'population-vector';
  readonly inputPopulationSize: number;

  private readonly preferredValues: Float32Array;
  private readonly windowMs: number;
  private readonly minOutput: number;
  private readonly maxOutput: number;
  private readonly spikeHistory: Array<[number, number]> = []; // [neuronIndex, time]

  constructor(config: PopulationVectorConfig) {
    this.inputPopulationSize = config.n;
    this.preferredValues = config.preferredValues;
    this.windowMs = config.windowMs ?? 30;
    this.minOutput = config.minOutput ?? -1;
    this.maxOutput = config.maxOutput ?? 1;
  }

  decode(spikeIndices: readonly number[], t: number): Float32Array {
    // Add new spikes to history
    for (const idx of spikeIndices) {
      this.spikeHistory.push([idx, t]);
    }

    // Remove spikes outside the integration window
    const tMin = t - this.windowMs;
    while (this.spikeHistory.length > 0 && (this.spikeHistory[0]![1]) < tMin) {
      this.spikeHistory.shift();
    }

    // Population vector: weighted sum of preferred values by spike count
    let numerator = 0;
    let denominator = 0;

    // Count spikes per neuron in the window
    const spikeCounts = new Float32Array(this.inputPopulationSize);
    for (const [neuronIdx] of this.spikeHistory) {
      spikeCounts[neuronIdx] = (spikeCounts[neuronIdx] as number) + 1;
    }

    for (let i = 0; i < this.inputPopulationSize; i++) {
      const rate = spikeCounts[i] as number;
      const pref = this.preferredValues[i] as number;
      numerator += rate * pref;
      denominator += rate;
    }

    const rawOutput = denominator > 0 ? numerator / denominator : 0;

    // Normalize to output range
    const prefMin = Math.min(...Array.from(this.preferredValues));
    const prefMax = Math.max(...Array.from(this.preferredValues));
    const prefRange = prefMax - prefMin;

    let normalized = prefRange > 0
      ? (rawOutput - prefMin) / prefRange
      : 0.5;

    const output = this.minOutput + normalized * (this.maxOutput - this.minOutput);
    return new Float32Array([Math.max(this.minOutput, Math.min(this.maxOutput, output))]);
  }
}

/**
 * Winner-takes-all decoder for discrete action selection.
 * The neuron with the highest recent firing rate selects the action.
 * Uses a sliding window; does NOT use softmax or any gradient-based selection.
 */
export class WTADecoder implements SpikeDecoder {
  readonly name = 'winner-takes-all';
  readonly inputPopulationSize: number;

  private readonly windowMs: number;
  private readonly spikeHistory: Array<[number, number]> = [];

  constructor(n: number, windowMs: number = 30) {
    this.inputPopulationSize = n;
    this.windowMs = windowMs;
  }

  decode(spikeIndices: readonly number[], t: number): Float32Array {
    for (const idx of spikeIndices) {
      this.spikeHistory.push([idx, t]);
    }

    const tMin = t - this.windowMs;
    while (this.spikeHistory.length > 0 && (this.spikeHistory[0]![1]) < tMin) {
      this.spikeHistory.shift();
    }

    const counts = new Float32Array(this.inputPopulationSize);
    for (const [idx] of this.spikeHistory) {
      counts[idx] = (counts[idx] as number) + 1;
    }

    // Find winner
    let maxCount = 0;
    let winner = 0;
    for (let i = 0; i < this.inputPopulationSize; i++) {
      if ((counts[i] as number) > maxCount) {
        maxCount = counts[i] as number;
        winner = i;
      }
    }

    // Return one-hot vector of winner
    const result = new Float32Array(this.inputPopulationSize);
    if (maxCount > 0) result[winner] = 1.0;
    return result;
  }
}
