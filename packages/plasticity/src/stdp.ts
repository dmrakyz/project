/**
 * Spike-Timing Dependent Plasticity (STDP) — nearest-neighbor formulation.
 *
 * The foundational Hebbian learning rule. Without STDP, the network cannot
 * self-organize based on experience. This is the primary synaptic modification
 * mechanism in V1.
 *
 * Nearest-neighbor STDP formulation:
 *   - Pre-synaptic trace r_i(t): decays as dr/dt = -r/τ_pre, +1 on pre-spike
 *   - Post-synaptic trace o_j(t): decays as do/dt = -o/τ_post, +1 on post-spike
 *
 *   When pre-synaptic neuron i fires:
 *     ΔW_ij = -A- × o_j(t)    (LTD: post fired before pre → weaken connection)
 *
 *   When post-synaptic neuron j fires:
 *     ΔW_ij = +A+ × r_i(t)    (LTP: pre fired before post → strengthen connection)
 *
 * This "nearest-neighbor" formulation considers only the most recent spike pair,
 * which is computationally tractable and experimentally motivated.
 *
 * Stability note: STDP alone leads to runaway dynamics. Homeostatic plasticity
 * (synaptic scaling) MUST run concurrently to bound weight growth. The timescale
 * of homeostasis must be ≥100× slower than STDP.
 *
 * Reference: van Rossum et al. (2000), Bi & Poo (1998, 2001).
 */

import type {
  PlasticityRule,
  PlasticityContext,
  WeightUpdate,
  STDPConfig,
} from '@snn/types';
import { NO_UPDATE, DEFAULT_STDP_CONFIG } from '@snn/types';

export class STDPRule implements PlasticityRule {
  readonly ruleId = 'stdp-nearest-neighbor';
  readonly description = 'Nearest-neighbor STDP: LTP on pre→post, LTD on post→pre';
  readonly requiredTraces = ['pre', 'post'] as const;

  private readonly config: Readonly<STDPConfig>;

  constructor(config: Partial<STDPConfig> = {}) {
    this.config = { ...DEFAULT_STDP_CONFIG, ...config };
  }

  /**
   * Called when a pre-synaptic neuron fires.
   * Implements LTD: post-synaptic trace o_j captures recent post-synaptic activity.
   * If post fired recently (high o_j), pre→post ordering means post fired BEFORE pre → LTD.
   */
  onPreSpike(
    _synapseIndex: number,
    _preTrace: number,
    postTrace: number,
    _eligibility: number,
    currentWeight: number,
    _context: PlasticityContext
  ): WeightUpdate {
    // LTD: ΔW = -A- × o_j(t)
    const delta = -this.config.aMinus * postTrace;
    return delta !== 0 ? { delta } : NO_UPDATE;
  }

  /**
   * Called when a post-synaptic neuron fires.
   * Implements LTP: pre-synaptic trace r_i captures recent pre-synaptic activity.
   * If pre fired recently (high r_i), pre→post ordering means pre fired BEFORE post → LTP.
   */
  onPostSpike(
    _synapseIndex: number,
    preTrace: number,
    _postTrace: number,
    _eligibility: number,
    currentWeight: number,
    _context: PlasticityContext
  ): WeightUpdate {
    // LTP: ΔW = +A+ × r_i(t)
    const delta = this.config.aPlus * preTrace;
    return delta !== 0 ? { delta } : NO_UPDATE;
  }

  getConfig(): Readonly<STDPConfig> {
    return this.config;
  }
}

/**
 * Manages the exponential decay of pre- and post-synaptic traces.
 * Called once per timestep by the plasticity engine.
 *
 * Trace update: x(t+dt) = x(t) × exp(-dt/tau)
 * (exact solution, not Euler approximation)
 */
export function decayTraces(
  preTraces: Float32Array,
  postTraces: Float32Array,
  dt: number,
  tauPre: number,
  tauPost: number
): void {
  const decayPre  = Math.exp(-dt / tauPre);
  const decayPost = Math.exp(-dt / tauPost);

  for (let i = 0; i < preTraces.length; i++) {
    preTraces[i] = (preTraces[i] as number) * decayPre;
  }
  for (let j = 0; j < postTraces.length; j++) {
    postTraces[j] = (postTraces[j] as number) * decayPost;
  }
}

/** Increment pre-synaptic trace for all outgoing synapses of a spiking neuron */
export function incrementPreTrace(
  preTraces: Float32Array,
  outgoingIndices: number[]
): void {
  for (const idx of outgoingIndices) {
    preTraces[idx] = (preTraces[idx] as number) + 1.0;
  }
}

/** Increment post-synaptic trace for a spiking target neuron */
export function incrementPostTrace(postTraces: Float32Array, targetIndex: number): void {
  postTraces[targetIndex] = (postTraces[targetIndex] as number) + 1.0;
}
