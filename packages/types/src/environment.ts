/**
 * Environment integration types — the interface between the neural simulation and the world.
 *
 * Design contract:
 * - SensorInput contains raw continuous data; encoding to spikes happens in packages/environment
 * - Motor output is always a spike population vector; decoding to actions happens in packages/environment
 * - ValenceSignal is the only path from the environment to the plasticity system (no direct weight access)
 * - No ANN shortcuts: no linear readout layers, no rate-times-weight operations
 */

/** Specification of a sensory channel */
export interface SensorSpec {
  readonly name: string;
  readonly dimensions: readonly number[];  // Shape of the sensory field (e.g. [28, 28] for grid)
  readonly minValue: number;
  readonly maxValue: number;
  readonly unit: string;
}

/** Specification of a motor output channel */
export interface ActuatorSpec {
  readonly name: string;
  readonly dimensions: readonly number[];
  readonly minValue: number;
  readonly maxValue: number;
  readonly unit: string;
}

/** Raw sensory data from the environment — before spike encoding */
export interface SensorInput {
  /** Flat array of sensory values in channel-major order */
  readonly data: Float32Array;
  /** Simulation time at which this observation was generated */
  readonly timestamp: number;
}

/**
 * Valence signal from the environment — maps to neuromodulator concentrations.
 * This is the only pathway for reward/punishment signals to reach the plasticity layer.
 *
 * All values in [-1, 1] for reward, [0, 1] for unsigned signals.
 */
export interface ValenceSignal {
  /** Primary reward signal [-1, 1]. Positive = rewarding, negative = aversive */
  readonly reward: number;
  /** Novelty/salience [0, 1] — drives ACh and NE modulation */
  readonly salience: number;
  /** Arousal level [0, 1] — drives NE modulation */
  readonly arousal: number;
}

export const NEUTRAL_VALENCE: Readonly<ValenceSignal> = {
  reward: 0,
  salience: 0,
  arousal: 0.5,
};

/**
 * Motor output: a vector of spike rates per actuator neuron, estimated over
 * a sliding window. Decoded by population vector methods, not matrix multiply.
 */
export interface PopulationSpikeVector {
  /** Estimated firing rates in Hz per neuron in the motor population */
  readonly rates: Float32Array;
  /** Size of the estimation window in ms */
  readonly windowMs: number;
  /** Timestamp of the end of the estimation window */
  readonly timestamp: number;
}

/**
 * The environment interface — the world as seen by the SNN.
 *
 * The environment is a physics engine, not a training set. It produces
 * continuous sensory streams; the SNN consumes these as spike populations.
 * The loop is closed: SNN output → environment → SNN input → ...
 */
export interface Environment {
  readonly name: string;
  readonly sensorSpecs: readonly SensorSpec[];
  readonly actuatorSpecs: readonly ActuatorSpec[];

  /** Initialize and return the first observation */
  reset(seed?: number): SensorInput;

  /**
   * Advance the environment by one physics step.
   * @param motorOutput  Decoded motor commands from the SNN
   * @param dt           Simulation timestep in ms
   * @returns            New sensory observation
   */
  step(motorOutput: PopulationSpikeVector, dt: number): SensorInput;

  /**
   * Returns the current reward/valence signal.
   * Called after step() to get the valence for the just-completed action.
   */
  getValenceSignal(): ValenceSignal;

  /** True if the episode has ended (e.g., task succeeded or agent fell) */
  isDone(): boolean;
}

/**
 * Spike encoding: converts continuous sensory data to spike events.
 * All encoders inject spikes as current (not voltage clamps) into input populations.
 */
export interface SpikeEncoder {
  readonly name: string;
  readonly outputPopulationSize: number;

  /**
   * Convert sensor data to spike events for this timestep.
   * Returns indices of neurons that should receive current injection (positive current only).
   * The magnitude of current injection is determined by the encoder's tuning parameters.
   */
  encode(input: SensorInput, dt: number, t: number): EncoderOutput;
}

export interface EncoderOutput {
  /** Indices of neurons receiving current injection */
  readonly activeNeurons: Uint32Array;
  /** Current injection magnitudes in pA for each active neuron */
  readonly currents: Float32Array;
}

/**
 * Motor decoding: converts spike patterns to motor commands.
 * Uses population vector decoding only — no matrix multiply, no linear readout.
 */
export interface SpikeDecoder {
  readonly name: string;
  readonly inputPopulationSize: number;

  /**
   * Decode spike events to a motor command.
   * Integrates over a sliding window; the window slides automatically each call.
   */
  decode(spikes: readonly number[], t: number): Float32Array;
}
