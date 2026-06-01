/**
 * Population-level stepping — applies a NeuronModel to all neurons in a population.
 *
 * This is the hot path in V1 TypeScript mode. It is written for correctness and clarity.
 * When WASM is available (V2), this function is bypassed in favor of the WASM core.
 *
 * Memory layout: SoA (Structure of Arrays).
 * State matrix has stateVectorSize blocks of N values each:
 *   [V_0, V_1, ..., V_{N-1}, W_0, W_1, ..., W_{N-1}, ...]
 *
 * Params matrix has paramVectorSize blocks of N values each.
 *
 * This layout enables SIMD operations and GPU coalesced memory access.
 */

import type { NeuronModel, Population } from '@snn/types';

export interface StepPopulationResult {
  /** Indices of neurons that spiked this timestep */
  readonly spikeIndices: number[];
}

/**
 * Read neuron i's state slice from SoA state matrix.
 * Returns a view (no copy) — the caller must not hold this reference across a step.
 */
function readNeuronState(
  state: Float64Array,
  neuronIndex: number,
  n: number,
  stateSize: number
): Float64Array {
  const slice = new Float64Array(stateSize);
  for (let s = 0; s < stateSize; s++) {
    slice[s] = state[s * n + neuronIndex] as number;
  }
  return slice;
}

/** Write neuron i's state slice back into SoA state matrix */
function writeNeuronState(
  state: Float64Array,
  neuronIndex: number,
  n: number,
  slice: Float64Array
): void {
  for (let s = 0; s < slice.length; s++) {
    state[s * n + neuronIndex] = slice[s] as number;
  }
}

/** Read neuron i's parameter slice from SoA params matrix */
function readNeuronParams(
  params: Float64Array,
  neuronIndex: number,
  n: number,
  paramSize: number
): Float64Array {
  const slice = new Float64Array(paramSize);
  for (let p = 0; p < paramSize; p++) {
    slice[p] = params[p * n + neuronIndex] as number;
  }
  return slice;
}

/**
 * Step an entire population forward by one timestep.
 *
 * @param population  Population with SoA state and params
 * @param model       The neuron model (AdEx, LIF, etc.)
 * @param currents    Synaptic + external current per neuron in pA (length = N)
 * @param dt          Timestep in ms
 * @param t           Current simulation time in ms
 */
export function stepPopulation(
  population: Population,
  model: NeuronModel,
  currents: Float64Array,
  dt: number,
  t: number
): StepPopulationResult {
  const n = population.config.size;
  const stateSize = model.stateVectorSize;
  const paramSize = model.parameterVectorSize;
  const spikeIndices: number[] = [];

  for (let i = 0; i < n; i++) {
    const neuronState  = readNeuronState(population.state, i, n, stateSize);
    const neuronParams = readNeuronParams(population.params, i, n, paramSize);
    const refractoryRemaining = population.refractory[i] as number;
    const current = currents[i] as number;

    const result = model.step(neuronState, neuronParams, current, dt, t, refractoryRemaining);

    if (result.spiked) {
      model.resetState(neuronState, neuronParams);
      // Start refractory period
      const tRef = neuronParams[10] as number; // T_REF index (ADEX_PARAMS.T_REF = 10, LIF_PARAMS.T_REF = 5)
      population.refractory[i] = tRef;
      spikeIndices.push(i);
      population.spikeCounts[i] = (population.spikeCounts[i] as number) + 1;
    } else if (refractoryRemaining > 0) {
      population.refractory[i] = Math.max(0, refractoryRemaining - dt);
    }

    writeNeuronState(population.state, i, n, neuronState);
  }

  return { spikeIndices };
}

/**
 * Initialize a population's SoA state matrix from the neuron model's initState.
 * Params must already be populated (via SoA layout) before calling this.
 */
export function initializePopulationState(population: Population, model: NeuronModel): void {
  const n = population.config.size;
  const paramSize = model.parameterVectorSize;

  for (let i = 0; i < n; i++) {
    const neuronParams = readNeuronParams(population.params, i, n, paramSize);
    const initialState = model.initState(neuronParams);
    writeNeuronState(population.state, i, n, initialState);
  }
}
