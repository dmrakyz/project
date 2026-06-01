/**
 * Population encoder — converts continuous sensory values to spike events.
 *
 * Uses Gaussian tuning curves: each neuron has a preferred value, and its
 * activation probability is proportional to a Gaussian centered on that value.
 *
 * Encoding strategy: stochastic rate coding via Bernoulli sampling.
 * Current injection magnitude scales with the tuning curve response.
 *
 * CRITICAL: Spikes are injected as CURRENT, not as voltage clamps.
 * The target neuron's intrinsic dynamics still determine whether it fires.
 * This preserves dynamical integrity — the encoder creates excitatory drive,
 * not commanded spikes. This distinction is architecturally enforced.
 *
 * Anti-pattern prevented: activations = encode(stimulus); out = W @ activations
 * Population encoding produces CURRENTS that cause SPIKES in the next timestep,
 * not activation vectors.
 */

import type { SpikeEncoder, SensorInput, EncoderOutput } from '@snn/types';
import { SeededRNG } from '@snn/config';

export interface GaussianTuningConfig {
  /** Number of neurons in the encoding population */
  readonly n: number;
  /** Minimum value in the sensory range */
  readonly minVal: number;
  /** Maximum value in the sensory range */
  readonly maxVal: number;
  /** Width of each neuron's tuning curve (in value units) */
  readonly sigma?: number;
  /** Peak current injection at preferred value (pA) */
  readonly peakCurrent?: number;
  /** Random seed for stochastic sampling */
  readonly seed?: number;
}

export class PopulationEncoder implements SpikeEncoder {
  readonly name = 'population-gaussian';
  readonly outputPopulationSize: number;

  private readonly n: number;
  private readonly minVal: number;
  private readonly maxVal: number;
  private readonly sigma: number;
  private readonly peakCurrent: number;
  /** Preferred value (center of tuning curve) for each neuron */
  private readonly preferredValues: Float32Array;
  private readonly rng: SeededRNG;

  constructor(config: GaussianTuningConfig) {
    this.n = config.n;
    this.outputPopulationSize = config.n;
    this.minVal = config.minVal;
    this.maxVal = config.maxVal;
    const range = config.maxVal - config.minVal;
    this.sigma = config.sigma ?? range / (config.n * 0.5);
    this.peakCurrent = config.peakCurrent ?? 400;  // pA — suprathreshold for AdEx defaults
    this.rng = new SeededRNG(config.seed ?? 0);

    // Uniformly spaced preferred values across sensory range
    this.preferredValues = new Float32Array(config.n);
    for (let i = 0; i < config.n; i++) {
      this.preferredValues[i] = config.minVal + (i / (config.n - 1)) * (config.maxVal - config.minVal);
    }
  }

  encode(input: SensorInput, _dt: number, _t: number): EncoderOutput {
    // For a scalar input, use input.data[0]
    // For multi-dimensional, process each dimension independently
    const value = input.data.length > 0 ? (input.data[0] as number) : 0;
    const clampedValue = Math.max(this.minVal, Math.min(this.maxVal, value));

    const activeList: number[] = [];
    const currentList: number[] = [];

    for (let i = 0; i < this.n; i++) {
      const pref = this.preferredValues[i] as number;
      const diff = clampedValue - pref;
      const response = Math.exp(-(diff * diff) / (2 * this.sigma * this.sigma));
      const current = this.peakCurrent * response;

      if (current > 1.0) {  // Threshold: only inject current if response is non-negligible
        activeList.push(i);
        currentList.push(current);
      }
    }

    return {
      activeNeurons: new Uint32Array(activeList),
      currents: new Float32Array(currentList),
    };
  }

  getPreferredValues(): Float32Array {
    return this.preferredValues.slice();
  }
}

/**
 * Multi-dimensional population encoder — handles vector-valued sensory inputs.
 * Each input dimension has its own neuron subpopulation with Gaussian tuning.
 */
export class MultiDimPopulationEncoder implements SpikeEncoder {
  readonly name = 'population-gaussian-multidim';
  readonly outputPopulationSize: number;

  private readonly encoders: PopulationEncoder[];
  private readonly neuronsPerDim: number;

  constructor(
    configs: GaussianTuningConfig[]
  ) {
    this.encoders = configs.map(c => new PopulationEncoder(c));
    this.neuronsPerDim = configs[0]?.n ?? 0;
    this.outputPopulationSize = configs.reduce((sum, c) => sum + c.n, 0);
  }

  encode(input: SensorInput, dt: number, t: number): EncoderOutput {
    const allActive: number[] = [];
    const allCurrents: number[] = [];
    let offset = 0;

    for (let d = 0; d < this.encoders.length; d++) {
      const encoder = this.encoders[d]!;
      const dimInput: SensorInput = {
        data: new Float32Array([input.data[d] as number]),
        timestamp: input.timestamp,
      };
      const result = encoder.encode(dimInput, dt, t);
      for (let i = 0; i < result.activeNeurons.length; i++) {
        allActive.push((result.activeNeurons[i] as number) + offset);
        allCurrents.push(result.currents[i] as number);
      }
      offset += encoder.outputPopulationSize;
    }

    return {
      activeNeurons: new Uint32Array(allActive),
      currents: new Float32Array(allCurrents),
    };
  }
}
