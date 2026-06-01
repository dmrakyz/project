/**
 * NetworkBuilder — constructs populations and projections with validation.
 *
 * The builder enforces all architectural constraints at construction time:
 *   - Connectivity density limits
 *   - Minimum delay enforcement
 *   - Receptor type V1 constraints
 *   - Population size limits
 *
 * Usage:
 *   const builder = new NetworkBuilder({ dt: 0.1, seed: 42 });
 *   const popId = builder.addPopulation({ name: 'input', size: 100, modelId: MODEL_IDS.ADEX, role: 'excitatory' });
 *   builder.addProjection({ ... });
 *   const network = builder.build();
 */

import type {
  Population,
  Projection,
  PopulationConfig,
  ProjectionConfig,
} from '@snn/types';
import { getModel } from '@snn/neurons';
import { initializePopulationState } from '@snn/neurons';
import { SeededRNG } from '@snn/config';
import { assertValid, validatePopulation, validateProjection } from '@snn/config';
import { DEFAULT_DT } from '@snn/config';
import { buildCSRStore } from './csr-store';
import { CircularDelayLine } from './delay-line';

export interface NetworkBuilderConfig {
  readonly dt?: number;
  readonly seed?: number;
  /** Maximum axonal delay in ms — determines delay line buffer size */
  readonly maxDelayMs?: number;
}

export interface BuiltNetwork {
  readonly populations: ReadonlyMap<number, Population>;
  readonly projections: ReadonlyMap<number, Projection>;
  readonly dt: number;
}

export class NetworkBuilder {
  private readonly dt: number;
  private readonly maxDelayMs: number;
  private readonly rng: SeededRNG;
  private readonly populationConfigs: Map<number, PopulationConfig> = new Map();
  private readonly projectionConfigs: Map<number, ProjectionConfig> = new Map();
  private nextPopId = 0;
  private nextProjId = 0;

  constructor(config: NetworkBuilderConfig = {}) {
    this.dt = config.dt ?? DEFAULT_DT;
    this.maxDelayMs = config.maxDelayMs ?? 20.0;
    this.rng = new SeededRNG(config.seed ?? 42);
  }

  addPopulation(config: PopulationConfig): number {
    assertValid(validatePopulation(config), `Population "${config.name}"`);
    const id = this.nextPopId++;
    this.populationConfigs.set(id, config);
    return id;
  }

  addProjection(config: ProjectionConfig): number {
    const sourceConfig = this.populationConfigs.get(config.sourcePopulationId);
    const targetConfig = this.populationConfigs.get(config.targetPopulationId);

    if (!sourceConfig) {
      throw new Error(`Projection "${config.name}": source population ${config.sourcePopulationId} not found`);
    }
    if (!targetConfig) {
      throw new Error(`Projection "${config.name}": target population ${config.targetPopulationId} not found`);
    }

    assertValid(
      validateProjection(config, sourceConfig.size, targetConfig.size),
      `Projection "${config.name}"`
    );

    const id = this.nextProjId++;
    this.projectionConfigs.set(id, config);
    return id;
  }

  build(): BuiltNetwork {
    const populations = new Map<number, Population>();
    const projections = new Map<number, Projection>();

    // Build populations
    for (const [id, config] of this.populationConfigs) {
      const model = getModel(config.modelId);
      const n = config.size;

      // Initialize SoA parameter matrix with biologically-motivated defaults
      const defaultParams = model.defaultParams();
      const params = new Float64Array(model.parameterVectorSize * n);
      for (let i = 0; i < n; i++) {
        for (let p = 0; p < model.parameterVectorSize; p++) {
          // Heterogeneous initialization: ±5% Gaussian variation around defaults
          const base = defaultParams[p] as number;
          const noise = base === 0 ? 0 : this.rng.normal(base, Math.abs(base) * 0.05);
          params[p * n + i] = noise;
        }
      }

      const state = new Float64Array(model.stateVectorSize * n);
      const population: Population = {
        id,
        config,
        state,
        params,
        refractory: new Float64Array(n),
        spikeCounts: new Float64Array(n),
      };

      initializePopulationState(population, model);
      populations.set(id, population);
    }

    // Build projections
    for (const [id, config] of this.projectionConfigs) {
      const sourceConfig = this.populationConfigs.get(config.sourcePopulationId)!;
      const targetConfig = this.populationConfigs.get(config.targetPopulationId)!;

      const store = buildCSRStore(
        sourceConfig.size,
        targetConfig.size,
        config.connectivity,
        config.weights,
        config.delays,
        config.receptorType,
        this.rng
      );

      const S = store.synapseCount;
      const maxDelayMs = this.maxDelayMs;

      const projection: Projection = {
        id,
        config,
        store,
        preTrace: new Float32Array(S),
        postTrace: new Float32Array(targetConfig.size),
        eligibilityTrace: new Float32Array(S),
        delayLine: new CircularDelayLine(maxDelayMs, this.dt),
      };

      projections.set(id, projection);
    }

    return { populations, projections, dt: this.dt };
  }
}
