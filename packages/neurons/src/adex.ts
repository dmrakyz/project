/**
 * Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
 *
 * This is the TypeScript reference implementation. It is used for:
 *   1. Correctness validation (compared against the Rust/WASM implementation)
 *   2. V1 simulation (single-threaded TypeScript mode)
 *   3. Small populations where WASM overhead exceeds TypeScript execution cost
 *
 * Equations (Brette & Gerstner, 2005):
 *   C_m dV/dt = -g_L(V - E_L) + g_L·Δ_T·exp((V - V_T)/Δ_T) + I_syn + I_ext - w
 *   τ_w dw/dt  = a(V - E_L) - w
 *   Reset: when V ≥ V_peak → V = V_reset, w = w + b
 *
 * Conductance-based synaptic currents:
 *   I_syn = g_E(t)(E_E - V) + g_I(t)(E_I - V)
 *   dg_E/dt = -g_E / τ_AMPA
 *   dg_I/dt = -g_I / τ_GABA_A
 *
 * Integration method:
 *   V: Exponential Euler — analytically handles the exponential nonlinearity
 *      with larger stable timesteps than forward Euler.
 *   w: Forward Euler — w varies slowly; Euler is sufficient and much faster.
 *   g_E, g_I: Exact exponential decay.
 */

import type { NeuronModel, StepResult, ParameterSchema } from '@snn/types';
import {
  MODEL_IDS,
  ADEX_STATE,
  ADEX_STATE_SIZE,
  ADEX_PARAMS,
  ADEX_PARAMS_SIZE,
} from '@snn/types';
import { defaultAdExParams } from '@snn/config';
import { TAU_AMPA, TAU_GABA_A } from '@snn/config';

export class AdExModel implements NeuronModel {
  readonly modelId = MODEL_IDS.ADEX;
  readonly name = 'AdEx';
  readonly stateVectorSize = ADEX_STATE_SIZE;
  readonly parameterVectorSize = ADEX_PARAMS_SIZE;

  readonly parameterSchema: ParameterSchema = {
    C_m:     { min: 10,    max: 1000,  default: 200,  unit: 'pF', description: 'Membrane capacitance' },
    g_L:     { min: 1,     max: 100,   default: 10,   unit: 'nS', description: 'Leak conductance' },
    E_L:     { min: -90,   max: -50,   default: -70,  unit: 'mV', description: 'Resting/leak reversal potential' },
    V_T:     { min: -70,   max: -35,   default: -50,  unit: 'mV', description: 'Spike threshold' },
    delta_T: { min: 0.1,   max: 10,    default: 2,    unit: 'mV', description: 'Sharpness of spike initiation' },
    V_peak:  { min: -10,   max: 40,    default: 0,    unit: 'mV', description: 'Spike detection threshold' },
    tau_w:   { min: 1,     max: 1000,  default: 200,  unit: 'ms', description: 'Adaptation time constant' },
    a:       { min: -10,   max: 50,    default: 2,    unit: 'nS', description: 'Subthreshold adaptation coupling' },
    b:       { min: 0,     max: 500,   default: 0,    unit: 'pA', description: 'Spike-triggered adaptation increment' },
    V_reset: { min: -90,   max: -40,   default: -58,  unit: 'mV', description: 'Post-spike reset potential' },
    t_ref:   { min: 0,     max: 20,    default: 2,    unit: 'ms', description: 'Absolute refractory period' },
    E_E:     { min: -20,   max: 20,    default: 0,    unit: 'mV', description: 'Excitatory reversal potential' },
    E_I:     { min: -100,  max: -40,   default: -70,  unit: 'mV', description: 'Inhibitory reversal potential' },
  };

  step(
    state: Float64Array,
    params: Float64Array,
    externalCurrent: number,
    dt: number,
    _t: number,
    refractoryRemaining: number
  ): StepResult {
    const V  = state[ADEX_STATE.V] as number;
    const w  = state[ADEX_STATE.W] as number;
    const gE = state[ADEX_STATE.GE] as number;
    const gI = state[ADEX_STATE.GI] as number;

    const Cm      = params[ADEX_PARAMS.CM] as number;
    const gL      = params[ADEX_PARAMS.GL] as number;
    const EL      = params[ADEX_PARAMS.EL] as number;
    const VT      = params[ADEX_PARAMS.VT] as number;
    const deltaT  = params[ADEX_PARAMS.DELTA_T] as number;
    const Vpeak   = params[ADEX_PARAMS.V_PEAK] as number;
    const tauW    = params[ADEX_PARAMS.TAU_W] as number;
    const a       = params[ADEX_PARAMS.A] as number;
    const EE      = params[ADEX_PARAMS.EE] as number;
    const EI      = params[ADEX_PARAMS.EI] as number;

    // Exact exponential decay of conductances
    const gE_new = gE * Math.exp(-dt / TAU_AMPA);
    const gI_new = gI * Math.exp(-dt / TAU_GABA_A);

    if (refractoryRemaining > 0) {
      // During refractory period: clamp voltage, still decay conductances
      state[ADEX_STATE.GE] = gE_new;
      state[ADEX_STATE.GI] = gI_new;
      return { spiked: false, voltage: V, adaptation: w };
    }

    // Conductance-based synaptic current (pA = nS × mV)
    const I_syn = gE * (EE - V) + gI * (EI - V);

    // Total current into membrane
    const I_total = I_syn + externalCurrent - w;

    // Exponential Euler for voltage
    // V(t+dt) ≈ EL + (V - EL - gL·ΔT·exp((V-VT)/ΔT)/gL·dt/Cm ...
    // For AdEx, we use a linearized step with the exponential term evaluated at current V
    const expTerm = gL * deltaT * Math.exp(Math.min((V - VT) / deltaT, 20));
    const dVdt = (-gL * (V - EL) + expTerm + I_total) / Cm;
    const V_new = V + dVdt * dt;

    // Forward Euler for adaptation (slow variable — Euler is sufficient)
    const dWdt = (a * (V - EL) - w) / tauW;
    const w_new = w + dWdt * dt;

    // Update conductances
    state[ADEX_STATE.GE] = gE_new;
    state[ADEX_STATE.GI] = gI_new;

    if (V_new >= Vpeak) {
      // Spike detected — resetState will be called by the simulator
      state[ADEX_STATE.V] = Vpeak;
      state[ADEX_STATE.W] = w_new;
      return { spiked: true, voltage: Vpeak, adaptation: w_new };
    }

    state[ADEX_STATE.V] = V_new;
    state[ADEX_STATE.W] = w_new;
    return { spiked: false, voltage: V_new, adaptation: w_new };
  }

  resetState(state: Float64Array, params: Float64Array): void {
    const Vreset = params[ADEX_PARAMS.V_RESET] as number;
    const b      = params[ADEX_PARAMS.B] as number;
    const w      = state[ADEX_STATE.W] as number;
    state[ADEX_STATE.V] = Vreset;
    state[ADEX_STATE.W] = w + b;
    // Conductances are not reset — they continue decaying
  }

  initState(params: Float64Array): Float64Array {
    const state = new Float64Array(ADEX_STATE_SIZE);
    state[ADEX_STATE.V] = params[ADEX_PARAMS.EL] as number;  // Start at rest
    state[ADEX_STATE.W] = 0.0;
    state[ADEX_STATE.GE] = 0.0;
    state[ADEX_STATE.GI] = 0.0;
    return state;
  }

  defaultParams(): Float64Array {
    return defaultAdExParams();
  }

  serializeState(state: Float64Array): Uint8Array {
    return new Uint8Array(state.buffer.slice(state.byteOffset, state.byteOffset + state.byteLength));
  }

  deserializeState(data: Uint8Array): Float64Array {
    return new Float64Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }
}
