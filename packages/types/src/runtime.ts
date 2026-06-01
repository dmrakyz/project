/**
 * Runtime types — describes the interface to the WASM simulation core and
 * future WebGPU/Worker acceleration layers.
 *
 * V1: TypeScript-only simulation (correctness before performance).
 * V2: WASM core for hot-path neuron updates, Worker parallelism.
 * V2-V3: WebGPU for populations >10K neurons.
 */

/** Handle to a WASM-allocated population — opaque in TS, pointer in Rust */
export type PopulationHandle = number & { readonly __brand: 'PopulationHandle' };

/** Handle to a WASM-allocated synapse matrix */
export type SynapseHandle = number & { readonly __brand: 'SynapseHandle' };

/**
 * WASM simulation core interface — describes the Rust wasm-bindgen exports.
 * TypeScript simulation falls back to the TypeScript neuron implementations.
 *
 * Key constraint: step_population takes currents as input (not activations),
 * returns spike indices (not rate vectors). This is SNN-correct behavior.
 */
export interface WASMSimulationCore {
  /** Allocate a population in WASM memory */
  allocate_population(nNeurons: number, modelId: number): PopulationHandle;

  /** Free a population from WASM memory */
  free_population(handle: PopulationHandle): void;

  /** Write parameter matrix to WASM memory (SoA layout) */
  set_neuron_parameters(handle: PopulationHandle, params: Float64Array): void;

  /** Read current state matrix from WASM memory (SoA layout) */
  get_neuron_states(handle: PopulationHandle, out: Float64Array): void;

  /** Write current state matrix to WASM memory (used for restore from checkpoint) */
  set_neuron_states(handle: PopulationHandle, states: Float64Array): void;

  /**
   * Integrate one timestep for all neurons in the population.
   * @param handle       Population handle
   * @param dt           Timestep in ms
   * @param t            Current simulation time in ms
   * @param currents     Current injection per neuron in pA (length = nNeurons)
   * @param refractory   Remaining refractory time per neuron in ms
   * @returns            Number of spikes this step
   *
   * Spike indices are written to an internal buffer; call get_spike_indices() after.
   */
  step_population(
    handle: PopulationHandle,
    dt: number,
    t: number,
    currents: Float32Array,
    refractory: Float64Array
  ): number;

  /** Read spike indices from the internal buffer after step_population */
  get_spike_indices(handle: PopulationHandle, out: Uint32Array): void;

  /** Compute synaptic currents: for each spike, scatter conductance updates */
  compute_synaptic_currents(
    synapseHandle: SynapseHandle,
    spikeIndices: Uint32Array,
    targetVoltages: Float32Array,
    dt: number,
    out: Float32Array
  ): void;
}

/** Runtime mode — determines which simulation backend is used */
export type RuntimeMode = 'typescript' | 'wasm' | 'gpu';

/** Simulation clock — single source of truth for simulation time */
export interface SimulationClock {
  /** Current simulation time in ms */
  readonly t: number;
  /** Timestep size in ms */
  readonly dt: number;
  /** Total elapsed steps */
  readonly step: number;
  /** Advance clock by one timestep */
  tick(): void;
  /** Reset clock to t=0 */
  reset(): void;
}
