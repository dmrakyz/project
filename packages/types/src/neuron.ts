/**
 * Core neuron model interface — the most load-bearing abstraction in the system.
 * This interface must remain stable across V1→V2→V3. Any change breaks every
 * neuron implementation and the WASM bridge simultaneously.
 *
 * Design contract:
 * - NeuronModel is a pure mathematical object: no network topology, no plasticity
 * - State is a flat Float64Array indexed by well-known offsets
 * - Parameters are a flat Float64Array with per-neuron rows (heterogeneous populations)
 * - step() is a pure function — identical inputs always produce identical outputs
 */

export const MODEL_IDS = {
  LIF: 0,
  ADEX: 1,
  IZHIKEVICH: 2,  // V3+
} as const;

export type ModelId = typeof MODEL_IDS[keyof typeof MODEL_IDS];

export interface StepResult {
  readonly spiked: boolean;
  /** Membrane voltage after integration (mV) */
  readonly voltage: number;
  /** Adaptation current after integration (pA) */
  readonly adaptation: number;
}

export interface ParameterBounds {
  readonly min: number;
  readonly max: number;
  readonly default: number;
  readonly unit: string;
  readonly description: string;
}

export interface ParameterSchema {
  readonly [name: string]: ParameterBounds;
}

/**
 * A neuron model is a pure mathematical description of single-neuron dynamics.
 * Implementations: AdExModel, LIFModel (TypeScript reference) + corresponding Rust/WASM.
 */
export interface NeuronModel {
  readonly modelId: ModelId;
  readonly name: string;
  /** Number of f64 values in the state vector per neuron */
  readonly stateVectorSize: number;
  /** Number of f64 values in the parameter vector per neuron */
  readonly parameterVectorSize: number;
  readonly parameterSchema: ParameterSchema;

  /**
   * Integrate dynamics by one timestep.
   * @param state      Neuron's current state slice (stateVectorSize elements)
   * @param params     Neuron's parameter slice (parameterVectorSize elements)
   * @param current    Total injected current in pA (synaptic + external)
   * @param dt         Timestep in ms
   * @param t          Current simulation time in ms
   * @param refractoryRemaining  Remaining refractory period in ms (0 if not refractory)
   */
  step(
    state: Float64Array,
    params: Float64Array,
    current: number,
    dt: number,
    t: number,
    refractoryRemaining: number
  ): StepResult;

  /** Reset state variables after a spike, applying post-spike rules */
  resetState(state: Float64Array, params: Float64Array): void;

  /** Initialize state to resting conditions given parameters */
  initState(params: Float64Array): Float64Array;

  /** Create default parameter vector using schema defaults */
  defaultParams(): Float64Array;

  /** Serialize state for checkpointing */
  serializeState(state: Float64Array): Uint8Array;

  /** Restore state from checkpoint */
  deserializeState(data: Uint8Array): Float64Array;
}

/** State vector field offsets for AdEx — used by both TypeScript and Rust */
export const ADEX_STATE = {
  V: 0,   // Membrane voltage (mV)
  W: 1,   // Adaptation current (pA)
  GE: 2,  // Excitatory conductance (nS)
  GI: 3,  // Inhibitory conductance (nS)
} as const;

export const ADEX_STATE_SIZE = 4;

/** Parameter vector field offsets for AdEx */
export const ADEX_PARAMS = {
  CM: 0,       // Membrane capacitance (pF)
  GL: 1,       // Leak conductance (nS)
  EL: 2,       // Leak reversal potential (mV)
  VT: 3,       // Spike threshold (mV)
  DELTA_T: 4,  // Slope factor (mV)
  V_PEAK: 5,   // Spike detection threshold (mV)
  TAU_W: 6,    // Adaptation time constant (ms)
  A: 7,        // Subthreshold adaptation coupling (nS)
  B: 8,        // Spike-triggered adaptation increment (pA)
  V_RESET: 9,  // Reset potential (mV)
  T_REF: 10,   // Absolute refractory period (ms)
  EE: 11,      // Excitatory reversal potential (mV)
  EI: 12,      // Inhibitory reversal potential (mV)
} as const;

export const ADEX_PARAMS_SIZE = 13;

/** State vector field offsets for LIF */
export const LIF_STATE = {
  V: 0,
  GE: 1,
  GI: 2,
} as const;

export const LIF_STATE_SIZE = 3;

export const LIF_PARAMS = {
  CM: 0,
  GL: 1,
  EL: 2,
  VT: 3,
  V_RESET: 4,
  T_REF: 5,
  EE: 6,
  EI: 7,
} as const;

export const LIF_PARAMS_SIZE = 8;
