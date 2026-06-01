/**
 * CartPole-analog environment — classic continuous control test for the SNN.
 *
 * A pole is balanced on a cart. The agent receives a 4D sensory observation
 * (cart position, cart velocity, pole angle, pole angular velocity) and produces
 * a continuous force on the cart.
 *
 * This is the primary V1 validation environment. Success criterion:
 * the SNN should keep the pole balanced for ≥200 timesteps purely through
 * STDP + homeostasis without any gradient descent.
 *
 * Physics: standard cartpole dynamics (Barto et al., 1983).
 *
 * Reward structure: +1 for every timestep the pole remains upright.
 * Episode ends when: |angle| > 12° or |position| > 2.4m.
 *
 * Sensory encoding: 4D state → 4 subpopulations via population encoder.
 * Motor decoding: 1 motor population (left force < 0, right force > 0).
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

const GRAVITY = 9.8;        // m/s²
const MASS_CART = 1.0;      // kg
const MASS_POLE = 0.1;      // kg
const POLE_LENGTH = 0.5;    // m (half-length)
const FORCE_MAGNITUDE = 10; // N (max force)
const MAX_ANGLE = 12 * (Math.PI / 180);   // radians
const MAX_POSITION = 2.4;   // meters
const DT_PHYSICS = 0.02;    // s (physics update rate — separate from neural dt)

export class CartPoleAnalog implements Environment {
  readonly name = 'cartpole-analog';

  readonly sensorSpecs: readonly SensorSpec[] = [
    { name: 'cart_position', dimensions: [1], minValue: -2.4, maxValue: 2.4, unit: 'm' },
    { name: 'cart_velocity', dimensions: [1], minValue: -4.0, maxValue: 4.0, unit: 'm/s' },
    { name: 'pole_angle',    dimensions: [1], minValue: -MAX_ANGLE, maxValue: MAX_ANGLE, unit: 'rad' },
    { name: 'pole_velocity', dimensions: [1], minValue: -4.0, maxValue: 4.0, unit: 'rad/s' },
  ];

  readonly actuatorSpecs: readonly ActuatorSpec[] = [
    { name: 'cart_force', dimensions: [1], minValue: -FORCE_MAGNITUDE, maxValue: FORCE_MAGNITUDE, unit: 'N' },
  ];

  private state: [number, number, number, number] = [0, 0, 0, 0];
  private t = 0;
  private done = false;
  private lastReward = 0;
  private rng: SeededRNG;

  constructor(seed?: number) {
    this.rng = new SeededRNG(seed ?? 42);
  }

  reset(seed?: number): SensorInput {
    if (seed !== undefined) this.rng = new SeededRNG(seed);
    // Small random initial perturbation
    this.state = [
      this.rng.uniform(-0.05, 0.05),
      this.rng.uniform(-0.05, 0.05),
      this.rng.uniform(-0.05, 0.05),
      this.rng.uniform(-0.05, 0.05),
    ];
    this.t = 0;
    this.done = false;
    this.lastReward = 0;
    return this.stateToSensorInput();
  }

  step(motorOutput: PopulationSpikeVector, _dt: number): SensorInput {
    if (this.done) return this.stateToSensorInput();

    // Decode motor output: rates[0] > rates[1] → push right; rates[1] > rates[0] → push left
    let force = 0;
    if (motorOutput.rates.length >= 2) {
      const rightDrive = motorOutput.rates[0] as number;
      const leftDrive = motorOutput.rates[1] as number;
      force = Math.tanh(rightDrive - leftDrive) * FORCE_MAGNITUDE;
    } else if (motorOutput.rates.length === 1) {
      force = Math.tanh((motorOutput.rates[0] as number) / 10) * FORCE_MAGNITUDE;
    }

    // Clamp force
    force = Math.max(-FORCE_MAGNITUDE, Math.min(FORCE_MAGNITUDE, force));

    // Physics integration (fixed sub-step for numerical stability)
    const [x, xDot, theta, thetaDot] = this.state;

    const totalMass = MASS_CART + MASS_POLE;
    const massPolePoleLen = MASS_POLE * POLE_LENGTH;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    const temp = (force + massPolePoleLen * thetaDot * thetaDot * sinTheta) / totalMass;
    const thetaAcc =
      (GRAVITY * sinTheta - cosTheta * temp) /
      (POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * cosTheta * cosTheta / totalMass));
    const xAcc = temp - massPolePoleLen * thetaAcc * cosTheta / totalMass;

    this.state = [
      x    + DT_PHYSICS * xDot,
      xDot + DT_PHYSICS * xAcc,
      theta    + DT_PHYSICS * thetaDot,
      thetaDot + DT_PHYSICS * thetaAcc,
    ];
    this.t += DT_PHYSICS;

    const [xNew, , thetaNew] = this.state;
    if (Math.abs(xNew) > MAX_POSITION || Math.abs(thetaNew) > MAX_ANGLE) {
      this.done = true;
      this.lastReward = -1.0;
    } else {
      this.lastReward = 1.0;
    }

    return this.stateToSensorInput();
  }

  getValenceSignal(): ValenceSignal {
    return {
      reward: this.lastReward,
      salience: Math.abs(this.state[2] as number) / MAX_ANGLE, // Pole angle urgency
      arousal: 0.5,
    };
  }

  isDone(): boolean {
    return this.done;
  }

  getState(): readonly [number, number, number, number] {
    return this.state;
  }

  private stateToSensorInput(): SensorInput {
    const [x, xDot, theta, thetaDot] = this.state;
    return {
      data: new Float32Array([x, xDot, theta, thetaDot]),
      timestamp: this.t * 1000, // Convert s → ms
    };
  }
}
