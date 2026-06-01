/**
 * RandomSpike environment — testing environment for verifying network dynamics.
 *
 * Generates random sensory input at a configurable rate.
 * Used for: verifying homeostatic stability, testing STDP convergence without
 * a meaningful task structure, and benchmarking simulation performance.
 *
 * No task — no reward. Valence signal is always neutral.
 */

import type { Environment, SensorInput, SensorSpec, ActuatorSpec, ValenceSignal, PopulationSpikeVector } from '@snn/types';
import { NEUTRAL_VALENCE } from '@snn/types';
import { SeededRNG } from '@snn/config';

export interface RandomSpikeConfig {
  readonly sensorDimensions: number;
  readonly actuatorDimensions: number;
  /** Mean firing rate of the random input in Hz */
  readonly inputRate?: number;
  readonly seed?: number;
}

export class RandomSpikeEnvironment implements Environment {
  readonly name = 'random-spike';
  readonly sensorSpecs: readonly SensorSpec[];
  readonly actuatorSpecs: readonly ActuatorSpec[];

  private readonly inputRate: number;
  private readonly rng: SeededRNG;
  private currentInput: SensorInput;

  constructor(config: RandomSpikeConfig) {
    this.inputRate = config.inputRate ?? 10;
    this.rng = new SeededRNG(config.seed ?? 42);

    this.sensorSpecs = Array.from({ length: config.sensorDimensions }, (_, i) => ({
      name: `sensor_${i}`,
      dimensions: [1] as const,
      minValue: 0,
      maxValue: 1,
      unit: 'normalized',
    }));

    this.actuatorSpecs = Array.from({ length: config.actuatorDimensions }, (_, i) => ({
      name: `actuator_${i}`,
      dimensions: [1] as const,
      minValue: -1,
      maxValue: 1,
      unit: 'normalized',
    }));

    this.currentInput = this.generateInput(0);
  }

  reset(_seed?: number): SensorInput {
    this.currentInput = this.generateInput(0);
    return this.currentInput;
  }

  step(_motorOutput: PopulationSpikeVector, dt: number): SensorInput {
    const t = this.currentInput.timestamp + dt;
    this.currentInput = this.generateInput(t);
    return this.currentInput;
  }

  getValenceSignal(): ValenceSignal {
    return NEUTRAL_VALENCE;
  }

  isDone(): boolean {
    return false;  // Never terminates
  }

  private generateInput(t: number): SensorInput {
    const data = new Float32Array(this.sensorSpecs.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = this.rng.random();
    }
    return { data, timestamp: t };
  }
}
