/**
 * Homeostatic plasticity — synaptic scaling and intrinsic excitability.
 *
 * Without homeostasis, STDP leads to catastrophic runaway dynamics:
 * networks either silence completely or saturate (all neurons firing at max rate).
 * Homeostasis is not optional — it must run concurrently with STDP from V1.
 *
 * Two mechanisms:
 *
 * 1. Synaptic scaling (Turrigiano et al., 1998):
 *    All incoming weights to a neuron are multiplied by a scaling factor.
 *    If firing rate > target: scale DOWN (reduce incoming drive)
 *    If firing rate < target: scale UP (increase incoming drive)
 *    Critical: scaling is MULTIPLICATIVE, not additive.
 *    This preserves relative weight structure while adjusting overall magnitude.
 *
 * 2. Intrinsic excitability (Zhang & Linden, 2003):
 *    Adjust V_T (spike threshold) per neuron based on recent activity.
 *    If firing too fast: raise V_T (harder to spike)
 *    If firing too slow: lower V_T (easier to spike)
 *    This is complementary to synaptic scaling — acts on intrinsic dynamics.
 *
 * CRITICAL TIMESCALE WARNING:
 * Homeostasis timescale must be ≥100× longer than STDP timescale.
 * If homeostasis runs at the same timescale as STDP, it erases learned structure.
 * Default: updateIntervalSteps = 10000 (1000ms at dt=0.1ms)
 * STDP τ ≈ 20ms → 200 steps → homeostasis runs 50× slower by default.
 * Researchers should consider increasing to 50000+ steps for long experiments.
 */

import type {
  HomeostaticConfig,
  HomeostaticState,
  Population,
  Projection,
} from '@snn/types';
import { DEFAULT_HOMEOSTATIC_CONFIG, ADEX_PARAMS, ADEX_PARAMS_SIZE } from '@snn/types';

export class HomeostaticPlasticity {
  private readonly config: Readonly<HomeostaticConfig>;
  private stepCounter: number = 0;

  /** Spike count accumulated since last homeostatic update — per neuron per population */
  private readonly accumulatedSpikes: Map<number, Float64Array> = new Map();

  constructor(config: Partial<HomeostaticConfig> = {}) {
    this.config = { ...DEFAULT_HOMEOSTATIC_CONFIG, ...config };
  }

  /** Record spikes for homeostasis rate estimation */
  recordSpikes(populationId: number, spikeIndices: number[], populationSize: number): void {
    if (!this.accumulatedSpikes.has(populationId)) {
      this.accumulatedSpikes.set(populationId, new Float64Array(populationSize));
    }
    const counts = this.accumulatedSpikes.get(populationId)!;
    for (const idx of spikeIndices) {
      counts[idx] = (counts[idx] as number) + 1;
    }
  }

  /**
   * Apply homeostatic updates if the update interval has elapsed.
   *
   * @param populations  All populations in the network
   * @param projections  All projections (for synaptic scaling access)
   * @param dt           Simulation timestep in ms
   * @returns            HomeostaticState per population (for observability)
   */
  onTimestep(
    populations: ReadonlyMap<number, Population>,
    projections: ReadonlyMap<number, Projection>,
    dt: number
  ): Map<number, HomeostaticState> | null {
    this.stepCounter++;
    if (this.stepCounter < this.config.updateIntervalSteps) {
      return null;
    }
    this.stepCounter = 0;

    const windowMs = this.config.updateIntervalSteps * dt;
    const results = new Map<number, HomeostaticState>();

    for (const [popId, population] of populations) {
      const n = population.config.size;
      const accumulated = this.accumulatedSpikes.get(popId) ?? new Float64Array(n);

      // Estimate firing rates (Hz) over the elapsed window
      const currentRates = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        currentRates[i] = ((accumulated[i] as number) / windowMs) * 1000;
      }

      const scalingFactors = new Float32Array(n).fill(1.0);
      const excitabilityOffsets = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const rate = currentRates[i] as number;
        const target = this.config.targetRate;
        const rateError = rate - target;

        // Synaptic scaling: multiplicative factor
        // factor < 1 when firing too fast (scale down inputs)
        // factor > 1 when firing too slow (scale up inputs)
        const factor = 1.0 - this.config.scalingStrength * rateError / target;
        scalingFactors[i] = Math.max(0.5, Math.min(2.0, factor));

        // Intrinsic excitability: shift V_T
        // Raise threshold if too fast, lower if too slow
        excitabilityOffsets[i] = this.config.excitabilityStrength * rateError;
      }

      // Apply synaptic scaling to all incoming projections
      this.applyScaling(popId, scalingFactors, projections);

      // Apply intrinsic excitability adjustment to V_T parameter
      this.applyExcitabilityShift(population, excitabilityOffsets);

      results.set(popId, {
        targetRate: this.config.targetRate,
        currentRates,
        scalingFactors,
        excitabilityOffsets,
      });

      // Reset accumulated spike counts
      if (this.accumulatedSpikes.has(popId)) {
        this.accumulatedSpikes.get(popId)!.fill(0);
      }
    }

    return results;
  }

  private applyScaling(
    targetPopId: number,
    scalingFactors: Float32Array,
    projections: ReadonlyMap<number, Projection>
  ): void {
    for (const proj of projections.values()) {
      if (proj.config.targetPopulationId !== targetPopId) continue;
      if (proj.config.plasticityRuleId === null) continue; // Don't scale fixed projections

      proj.store.forEach(syn => {
        const factor = scalingFactors[syn.targetIndex] as number;
        syn.weight *= factor;
      });
      proj.store.clampWeights(0, 10);
    }
  }

  private applyExcitabilityShift(population: Population, offsets: Float32Array): void {
    const n = population.config.size;
    const paramSize = population.params.length / n;

    // AdEx: V_T is at index ADEX_PARAMS.VT = 3
    // LIF: V_T is at index LIF_PARAMS.VT = 3
    // Both happen to be at index 3 — access via well-known offset
    const VT_PARAM_INDEX = 3;
    if (paramSize < VT_PARAM_INDEX + 1) return;

    for (let i = 0; i < n; i++) {
      const paramIdx = VT_PARAM_INDEX * n + i;
      const currentVT = population.params[paramIdx] as number;
      const offset = offsets[i] as number;
      // Clamp within ±10mV of the neuron's base threshold
      population.params[paramIdx] = Math.max(
        currentVT - 5,
        Math.min(currentVT + 5, currentVT + offset)
      );
    }
  }
}
