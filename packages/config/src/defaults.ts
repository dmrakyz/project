/**
 * Biologically-motivated default parameters.
 *
 * These defaults are derived from experimental measurements in cortical neurons.
 * They are starting points for parameter sweeps, not fixed values.
 * All units follow the SI-adjacent convention: mV, ms, pA, nS, pF.
 */

import {
  ADEX_PARAMS_SIZE,
  ADEX_PARAMS,
  LIF_PARAMS_SIZE,
  LIF_PARAMS,
} from '@snn/types';

/**
 * AdEx default parameters for a regular spiking cortical excitatory neuron.
 * Source: Brette & Gerstner (2005), Naud et al. (2008).
 */
export function defaultAdExParams(): Float64Array {
  const p = new Float64Array(ADEX_PARAMS_SIZE);
  p[ADEX_PARAMS.CM] = 200.0;    // Membrane capacitance (pF)
  p[ADEX_PARAMS.GL] = 10.0;     // Leak conductance (nS)
  p[ADEX_PARAMS.EL] = -70.0;    // Resting potential (mV)
  p[ADEX_PARAMS.VT] = -50.0;    // Spike threshold (mV)
  p[ADEX_PARAMS.DELTA_T] = 2.0; // Slope factor (mV)
  p[ADEX_PARAMS.V_PEAK] = 0.0;  // Spike detection threshold (mV)
  p[ADEX_PARAMS.TAU_W] = 200.0; // Adaptation time constant (ms)
  p[ADEX_PARAMS.A] = 2.0;       // Subthreshold adaptation (nS)
  p[ADEX_PARAMS.B] = 0.0;       // Spike-triggered adaptation (pA) — 0 = no bursting
  p[ADEX_PARAMS.V_RESET] = -58.0; // Reset potential (mV)
  p[ADEX_PARAMS.T_REF] = 2.0;   // Absolute refractory period (ms)
  p[ADEX_PARAMS.EE] = 0.0;      // Excitatory reversal (mV)
  p[ADEX_PARAMS.EI] = -70.0;    // Inhibitory reversal (mV)
  return p;
}

/**
 * AdEx parameters for a fast-spiking inhibitory interneuron.
 * Faster membrane time constant, lower adaptation.
 */
export function fastSpikingAdExParams(): Float64Array {
  const p = defaultAdExParams();
  p[ADEX_PARAMS.CM] = 100.0;
  p[ADEX_PARAMS.GL] = 20.0;     // Faster leak → shorter time constant
  p[ADEX_PARAMS.VT] = -47.0;
  p[ADEX_PARAMS.DELTA_T] = 0.5; // Sharper threshold
  p[ADEX_PARAMS.TAU_W] = 100.0;
  p[ADEX_PARAMS.A] = 0.0;
  p[ADEX_PARAMS.B] = 0.0;
  p[ADEX_PARAMS.V_RESET] = -65.0;
  p[ADEX_PARAMS.T_REF] = 1.0;
  return p;
}

/**
 * AdEx parameters for a bursting excitatory neuron.
 * Large spike-triggered adaptation increment b drives bursts.
 */
export function burstingAdExParams(): Float64Array {
  const p = defaultAdExParams();
  p[ADEX_PARAMS.B] = 80.0;      // Large spike-triggered adaptation → burst termination
  p[ADEX_PARAMS.TAU_W] = 100.0;
  p[ADEX_PARAMS.V_RESET] = -46.0; // High reset → rapid re-spiking within burst
  return p;
}

/**
 * LIF default parameters (simpler model for testing and comparison).
 */
export function defaultLIFParams(): Float64Array {
  const p = new Float64Array(LIF_PARAMS_SIZE);
  p[LIF_PARAMS.CM] = 200.0;
  p[LIF_PARAMS.GL] = 10.0;
  p[LIF_PARAMS.EL] = -70.0;
  p[LIF_PARAMS.VT] = -50.0;
  p[LIF_PARAMS.V_RESET] = -65.0;
  p[LIF_PARAMS.T_REF] = 2.0;
  p[LIF_PARAMS.EE] = 0.0;
  p[LIF_PARAMS.EI] = -70.0;
  return p;
}

/** AMPA conductance decay time constant (ms) */
export const TAU_AMPA = 5.0;

/** GABA-A conductance decay time constant (ms) */
export const TAU_GABA_A = 10.0;

/** NMDA conductance rise time constant (ms) — V2 */
export const TAU_NMDA_RISE = 2.0;

/** NMDA conductance decay time constant (ms) — V2 */
export const TAU_NMDA_DECAY = 80.0;

/** Maximum connectivity density — enforced at projection construction */
export const MAX_WITHIN_POPULATION_CONNECTIVITY = 0.10;
export const MAX_BETWEEN_POPULATION_CONNECTIVITY = 0.05;

/** Default simulation timestep in ms */
export const DEFAULT_DT = 0.1;

/** Default target firing rate for homeostatic plasticity (Hz) */
export const DEFAULT_TARGET_RATE = 5.0;
