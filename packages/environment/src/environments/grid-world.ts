/**
 * GridWorld environment — discrete navigation task.
 *
 * A 2D grid where the agent navigates from a start to a goal position.
 * Demonstrates: spatial encoding, action selection via WTA, reward learning.
 *
 * Sensory encoding: 2D position → population code (normalized x, y).
 * Motor decoding: WTA over 4 action neurons (up, down, left, right).
 * Reward: +1.0 at goal, -0.01 per step (time pressure).
 */

import type {
  Environment,
  SensorInput,
  SensorSpec,
  ActuatorSpec,
  ValenceSignal,
  PopulationSpikeVector,
} from '@snn/types';
import { SeededRNG } from '@snn/config';

export interface GridWorldConfig {
  readonly width: number;
  readonly height: number;
  readonly maxSteps?: number;
  readonly seed?: number;
}

export class GridWorld implements Environment {
  readonly name = 'grid-world';

  readonly sensorSpecs: readonly SensorSpec[] = [
    { name: 'position_x', dimensions: [1], minValue: 0, maxValue: 1, unit: 'normalized' },
    { name: 'position_y', dimensions: [1], minValue: 0, maxValue: 1, unit: 'normalized' },
    { name: 'goal_x',     dimensions: [1], minValue: 0, maxValue: 1, unit: 'normalized' },
    { name: 'goal_y',     dimensions: [1], minValue: 0, maxValue: 1, unit: 'normalized' },
  ];

  readonly actuatorSpecs: readonly ActuatorSpec[] = [
    { name: 'action', dimensions: [4], minValue: 0, maxValue: 1, unit: 'one-hot' },
  ];

  private readonly width: number;
  private readonly height: number;
  private readonly maxSteps: number;
  private readonly rng: SeededRNG;

  private agentX = 0;
  private agentY = 0;
  private goalX = 0;
  private goalY = 0;
  private steps = 0;
  private done = false;
  private lastReward = 0;

  // Action vectors: [up, down, left, right]
  private static readonly ACTIONS = [
    [0, -1], [0, 1], [-1, 0], [1, 0],
  ] as const;

  constructor(config: GridWorldConfig) {
    this.width = config.width;
    this.height = config.height;
    this.maxSteps = config.maxSteps ?? config.width * config.height * 4;
    this.rng = new SeededRNG(config.seed ?? 42);
  }

  reset(_seed?: number): SensorInput {
    this.agentX = 0;
    this.agentY = 0;
    this.goalX = this.width - 1;
    this.goalY = this.height - 1;
    this.steps = 0;
    this.done = false;
    this.lastReward = 0;
    return this.makeSensorInput();
  }

  step(motorOutput: PopulationSpikeVector, _dt: number): SensorInput {
    if (this.done) return this.makeSensorInput();

    // Decode action: index of maximum rate in the 4-action population
    let action = 0;
    let maxRate = -Infinity;
    const n = Math.min(4, motorOutput.rates.length);
    for (let i = 0; i < n; i++) {
      if ((motorOutput.rates[i] as number) > maxRate) {
        maxRate = motorOutput.rates[i] as number;
        action = i;
      }
    }

    // Only move if any neuron actually fired
    if (maxRate > 0) {
      const [dx, dy] = GridWorld.ACTIONS[action]!;
      this.agentX = Math.max(0, Math.min(this.width - 1, this.agentX + dx));
      this.agentY = Math.max(0, Math.min(this.height - 1, this.agentY + dy));
    }

    this.steps++;

    if (this.agentX === this.goalX && this.agentY === this.goalY) {
      this.done = true;
      this.lastReward = 1.0;
    } else if (this.steps >= this.maxSteps) {
      this.done = true;
      this.lastReward = -0.01;
    } else {
      this.lastReward = -0.01;
    }

    return this.makeSensorInput();
  }

  getValenceSignal(): ValenceSignal {
    const distToGoal = Math.abs(this.agentX - this.goalX) + Math.abs(this.agentY - this.goalY);
    const maxDist = this.width + this.height - 2;
    return {
      reward: this.lastReward,
      salience: 1 - distToGoal / maxDist,
      arousal: 0.5,
    };
  }

  isDone(): boolean {
    return this.done;
  }

  private makeSensorInput(): SensorInput {
    return {
      data: new Float32Array([
        this.agentX / (this.width - 1),
        this.agentY / (this.height - 1),
        this.goalX / (this.width - 1),
        this.goalY / (this.height - 1),
      ]),
      timestamp: this.steps * 20, // 20ms per step
    };
  }
}
