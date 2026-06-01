/**
 * Network-level types: populations, projections, spike events, and connectivity.
 *
 * The ConnectivityStore interface is the load-bearing abstraction that allows
 * V3 structural plasticity (dynamic COO store) to replace the V1 CSR store
 * without touching any code above this boundary.
 */

import type { ModelId } from './neuron';

/** A spike event: a neuron fired at a specific simulation time */
export interface SpikeEvent {
  readonly neuronIndex: number;  // Local index within its population
  readonly populationId: number;
  readonly time: number;         // Simulation time in ms
}

/** Neuron role within the network — gates valid projection types */
export type NeuronRole = 'excitatory' | 'inhibitory' | 'modulatory' | 'input' | 'output';

/** A named group of neurons sharing the same model type and role */
export interface PopulationConfig {
  readonly name: string;
  readonly size: number;
  readonly modelId: ModelId;
  readonly role: NeuronRole;
  /** Optional spatial layout for distance-dependent connectivity */
  readonly geometry?: PopulationGeometry;
}

export interface PopulationGeometry {
  readonly type: '1d' | '2d' | '3d';
  /** Dimensions in micrometers */
  readonly dimensions: readonly number[];
}

/** Runtime population state — created by the simulator from config */
export interface Population {
  readonly id: number;
  readonly config: PopulationConfig;
  /** Flat SoA state matrix: [stateVectorSize × N] interleaved as [state0_n0, state0_n1, ..., state1_n0, ...] */
  state: Float64Array;
  /** Flat SoA parameter matrix: [paramVectorSize × N] */
  params: Float64Array;
  /** Remaining refractory time per neuron (ms) */
  refractory: Float64Array;
  /** Accumulated spike count per neuron (for homeostasis) */
  spikeCounts: Float64Array;
}

/** Synaptic receptor type */
export type ReceptorType = 'AMPA' | 'GABA_A' | 'NMDA' | 'GABA_B';

/** V1 includes AMPA and GABA_A; NMDA and GABA_B are V2 */
export const V1_RECEPTOR_TYPES: ReadonlySet<ReceptorType> = new Set(['AMPA', 'GABA_A']);

/**
 * ConnectivityStore is the abstraction boundary for synaptic connectivity.
 *
 * V1: CSRConnectivityStore — efficient for static/slowly-changing connectivity.
 * V3: DynamicConnectivityStore (COO-backed with CSR materialization for performance).
 *
 * Nothing above this interface knows which implementation is in use.
 */
export interface ConnectivityStore {
  readonly sourceSize: number;
  readonly targetSize: number;
  readonly synapseCount: number;

  /** Get all synapses originating from source neuron i */
  getOutgoing(sourceIndex: number): SynapseView[];

  /** Get all synapses targeting neuron j */
  getIncoming(targetIndex: number): SynapseView[];

  /** Iterate all synapses — used by plasticity rules */
  forEach(callback: (synapse: SynapseView) => void): void;

  /** Get weight for a specific synapse by index */
  getWeight(synapseIndex: number): number;

  /** Update weight for a specific synapse — core plasticity operation */
  setWeight(synapseIndex: number, weight: number): void;

  /** Clamp all weights to [minWeight, maxWeight] */
  clampWeights(minWeight: number, maxWeight: number): void;

  /** Get all weights as a view (may be a copy depending on implementation) */
  getWeightsSnapshot(): Float32Array;
}

/** A view of a single synapse — returned by ConnectivityStore queries */
export interface SynapseView {
  readonly index: number;       // Synapse index within the store
  readonly sourceIndex: number; // Source neuron local index
  readonly targetIndex: number; // Target neuron local index
  readonly delay: number;       // Propagation delay in ms
  readonly receptorType: ReceptorType;
  weight: number;               // Mutable — plasticity writes here
}

/** How connections are formed between populations */
export type ConnectivityRule =
  | { type: 'all_to_all' }
  | { type: 'one_to_one' }
  | { type: 'random'; probability: number; seed?: number }
  | { type: 'fixed_number_pre'; n: number; seed?: number }
  | { type: 'topographic'; sigma: number }  // Gaussian distance-dependent
  | { type: 'custom'; connections: ReadonlyArray<[number, number]> };

/** Configuration for creating a projection between two populations */
export interface ProjectionConfig {
  readonly name: string;
  readonly sourcePopulationId: number;
  readonly targetPopulationId: number;
  readonly receptorType: ReceptorType;
  readonly connectivity: ConnectivityRule;
  /** Initial weight distribution */
  readonly weights: WeightDistribution;
  /** Propagation delay in ms — minimum 0.1ms (one timestep at dt=0.1) */
  readonly delays: DelayDistribution;
  /** Plasticity rule ID — null means fixed (document the reason) */
  readonly plasticityRuleId: string | null;
  /** If null and plasticityRuleId is null, this should be explicitly documented */
  readonly plasticityRuleNote?: string;
}

export type WeightDistribution =
  | { type: 'constant'; value: number }
  | { type: 'uniform'; min: number; max: number }
  | { type: 'normal'; mean: number; std: number }
  | { type: 'lognormal'; mu: number; sigma: number };

export type DelayDistribution =
  | { type: 'constant'; value: number }
  | { type: 'uniform'; min: number; max: number };

/** A live projection: bidirectional reference between populations + connectivity */
export interface Projection {
  readonly id: number;
  readonly config: ProjectionConfig;
  readonly store: ConnectivityStore;
  /** Pre-synaptic trace per synapse — decays exponentially, increments on pre-spike */
  preTrace: Float32Array;
  /** Post-synaptic trace per target neuron */
  postTrace: Float32Array;
  /** Eligibility trace per synapse — accumulates STDP signal, gated by neuromodulator (V2) */
  eligibilityTrace: Float32Array;
  /** Delay-line buffer: circular buffer of spike event arrays, one slot per delay step */
  readonly delayLine: DelayLine;
}

/** Circular buffer implementing axonal conduction delays */
export interface DelayLine {
  readonly maxDelaySteps: number;
  writeHead: number;
  /** Fixed-size buffer: maxDelaySteps slots, each slot can hold multiple spikes */
  buffer: Array<number[]>;

  /** Record a spike to be delivered after `delaySteps` timesteps */
  enqueue(neuronIndex: number, delaySteps: number): void;

  /** Retrieve and clear spikes due at the current timestep */
  dequeue(): readonly number[];

  /** Advance the write head by one timestep */
  advance(): void;
}
