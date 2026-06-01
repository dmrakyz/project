/**
 * Seeded random number generation and biologically-motivated parameter distributions.
 *
 * Neuron parameter initialization draws from distributions matching experimental
 * measurements — NOT Xavier/He initialization. Those target gradient flow properties
 * irrelevant to SNN dynamics.
 *
 * Uses a linear congruential generator for reproducibility across platforms.
 */

export class SeededRNG {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed >>> 0;
  }

  /** Uniform random float in [0, 1) */
  random(): number {
    // LCG parameters from Numerical Recipes
    this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0;
    return this.state / 4294967296;
  }

  /** Uniform random float in [min, max) */
  uniform(min: number, max: number): number {
    return min + this.random() * (max - min);
  }

  /** Standard normal via Box-Muller transform */
  normal(mean: number = 0, std: number = 1): number {
    const u1 = this.random();
    const u2 = this.random();
    const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  /** Lognormal distribution — appropriate for synaptic weight distributions */
  lognormal(mu: number, sigma: number): number {
    return Math.exp(this.normal(mu, sigma));
  }

  /** Positive normal, clamped to min — for parameters that must be positive */
  positiveNormal(mean: number, cv: number = 0.1, min: number = mean * 0.1): number {
    return Math.max(min, this.normal(mean, mean * cv));
  }

  /** Random integer in [min, max] inclusive */
  randint(min: number, max: number): number {
    return Math.floor(this.uniform(min, max + 1));
  }

  /** Fisher-Yates shuffle of an array (in-place) */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randint(0, i);
      const tmp = arr[i] as T;
      arr[i] = arr[j] as T;
      arr[j] = tmp;
    }
    return arr;
  }
}

/**
 * Sample a weight distribution into a pre-allocated Float32Array.
 * All weight initializations go through this function — not through ANN init schemes.
 */
export function sampleWeights(
  dist: import('@snn/types').WeightDistribution,
  n: number,
  rng: SeededRNG
): Float32Array {
  const weights = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (dist.type) {
      case 'constant':
        weights[i] = dist.value;
        break;
      case 'uniform':
        weights[i] = rng.uniform(dist.min, dist.max);
        break;
      case 'normal':
        weights[i] = Math.max(0, rng.normal(dist.mean, dist.std));
        break;
      case 'lognormal':
        weights[i] = rng.lognormal(dist.mu, dist.sigma);
        break;
    }
  }
  return weights;
}

/**
 * Sample delays (in ms) from a delay distribution.
 * All delays are clamped to minimum 0.1ms (one timestep at dt=0.1ms).
 */
export function sampleDelays(
  dist: import('@snn/types').DelayDistribution,
  n: number,
  rng: SeededRNG,
  minDelay: number = 0.1
): Float32Array {
  const delays = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    switch (dist.type) {
      case 'constant':
        delays[i] = Math.max(minDelay, dist.value);
        break;
      case 'uniform':
        delays[i] = Math.max(minDelay, rng.uniform(dist.min, dist.max));
        break;
    }
  }
  return delays;
}
