/**
 * Leaky Integrate-and-Fire (LIF) neuron model — TypeScript reference implementation.
 *
 * LIF is included for:
 *   1. Baseline comparisons (simpler dynamics than AdEx)
 *   2. Fast simulation of large input populations where AdEx dynamics are unnecessary
 *   3. Testing the NeuronModel interface without AdEx complexity
 *
 * LIF cannot produce: bursting, adaptation, irregular firing, chattering.
 * AdEx is the primary model for any research on emergent dynamics.
 *
 * Equations:
 *   C_m dV/dt = -g_L(V - E_L) + I_syn + I_ext
 *   I_syn = g_E(t)(E_E - V) + g_I(t)(E_I - V)
 *   Reset: when V ≥ V_T → V = V_reset
 */

import type { NeuronModel, StepResult, ParameterSchema } from '@snn/types';
import { MODEL_IDS, LIF_STATE, LIF_STATE_SIZE, LIF_PARAMS, LIF_PARAMS_SIZE } from '@snn/types';
import { defaultLIFParams } from '@snn/config';
import { TAU_AMPA, TAU_GABA_A } from '@snn/config';

export class LIFModel implements NeuronModel {
  readonly modelId = MODEL_IDS.LIF;
  readonly name = 'LIF';
  readonly stateVectorSize = LIF_STATE_SIZE;
  readonly parameterVectorSize = LIF_PARAMS_SIZE;

  readonly parameterSchema: ParameterSchema = {
    C_m:     { min: 10,   max: 1000, default: 200,  unit: 'pF', description: 'Membrane capacitance' },
    g_L:     { min: 1,    max: 100,  default: 10,   unit: 'nS', description: 'Leak conductance' },
    E_L:     { min: -90,  max: -50,  default: -70,  unit: 'mV', description: 'Resting potential' },
    V_T:     { min: -70,  max: -35,  default: -50,  unit: 'mV', description: 'Spike threshold' },
    V_reset: { min: -90,  max: -40,  default: -65,  unit: 'mV', description: 'Reset potential' },
    t_ref:   { min: 0,    max: 20,   default: 2,    unit: 'ms', description: 'Refractory period' },
    E_E:     { min: -20,  max: 20,   default: 0,    unit: 'mV', description: 'Excitatory reversal' },
    E_I:     { min: -100, max: -40,  default: -70,  unit: 'mV', description: 'Inhibitory reversal' },
  };

  step(
    state: Float64Array,
    params: Float64Array,
    externalCurrent: number,
    dt: number,
    _t: number,
    refractoryRemaining: number
  ): StepResult {
    const V  = state[LIF_STATE.V] as number;
    const gE = state[LIF_STATE.GE] as number;
    const gI = state[LIF_STATE.GI] as number;

    const Cm     = params[LIF_PARAMS.CM] as number;
    const gL     = params[LIF_PARAMS.GL] as number;
    const EL     = params[LIF_PARAMS.EL] as number;
    const VT     = params[LIF_PARAMS.VT] as number;
    const EE     = params[LIF_PARAMS.EE] as number;
    const EI     = params[LIF_PARAMS.EI] as number;

    // Exact decay of conductances
    const gE_new = gE * Math.exp(-dt / TAU_AMPA);
    const gI_new = gI * Math.exp(-dt / TAU_GABA_A);

    state[LIF_STATE.GE] = gE_new;
    state[LIF_STATE.GI] = gI_new;

    if (refractoryRemaining > 0) {
      return { spiked: false, voltage: V, adaptation: 0 };
    }

    const I_syn = gE * (EE - V) + gI * (EI - V);
    const dVdt = (-gL * (V - EL) + I_syn + externalCurrent) / Cm;
    const V_new = V + dVdt * dt;

    if (V_new >= VT) {
      state[LIF_STATE.V] = VT;
      return { spiked: true, voltage: VT, adaptation: 0 };
    }

    state[LIF_STATE.V] = V_new;
    return { spiked: false, voltage: V_new, adaptation: 0 };
  }

  resetState(state: Float64Array, params: Float64Array): void {
    state[LIF_STATE.V] = params[LIF_PARAMS.V_RESET] as number;
  }

  initState(params: Float64Array): Float64Array {
    const state = new Float64Array(LIF_STATE_SIZE);
    state[LIF_STATE.V] = params[LIF_PARAMS.EL] as number;
    state[LIF_STATE.GE] = 0.0;
    state[LIF_STATE.GI] = 0.0;
    return state;
  }

  defaultParams(): Float64Array {
    return defaultLIFParams();
  }

  serializeState(state: Float64Array): Uint8Array {
    return new Uint8Array(state.buffer.slice(state.byteOffset, state.byteOffset + state.byteLength));
  }

  deserializeState(data: Uint8Array): Float64Array {
    return new Float64Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }
}
