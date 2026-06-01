/**
 * Single-threaded SNN simulator — V1 implementation.
 *
 * This is the main simulation loop. One timestep consists of:
 *   1. Advance clock
 *   2. Deliver queued spikes from delay lines → synaptic conductance updates
 *   3. Step neuron dynamics (AdEx/LIF integration)
 *   4. Record new spikes
 *   5. Enqueue new spikes into delay lines
 *   6. Apply plasticity (STDP traces + weight updates)
 *   7. Emit observability events
 *   8. (If environment is attached) encode new sensory input as currents
 *
 * Architectural invariants enforced here:
 *   - Spikes are never delivered at the same timestep they are generated (min 1 step delay)
 *   - Weight updates are applied via PlasticityEngine, never by direct matrix access
 *   - Sensory input enters as current injection, not as voltage clamp
 *   - Motor output is decoded from spike patterns, not from rate vectors multiplied by weights
 *
 * V2 will replace this with a multi-worker simulator that distributes
 * populations across Web Workers with SharedArrayBuffer spike buffers.
 */

import type { Population, Projection, Environment, SensorInput, PopulationSpikeVector } from '@snn/types';
import type { EventBus } from '@snn/types';
import { getModel } from '@snn/neurons';
import { stepPopulation } from '@snn/neurons';
import { deliverSpikes } from '@snn/network';
import { PlasticityEngine } from '@snn/plasticity';
import type { PlasticityEngineConfig } from '@snn/plasticity';
import { SimClock } from './clock';

export interface SimulatorConfig {
  readonly dt?: number;
  readonly plasticityConfig?: PlasticityEngineConfig;
}

export interface SimulationStep {
  readonly t: number;
  readonly step: number;
  /** Total spikes generated this step across all populations */
  readonly totalSpikes: number;
  /** Spikes per population: Map<populationId, spikeIndices> */
  readonly spikesByPopulation: ReadonlyMap<number, readonly number[]>;
}

export class Simulator {
  private readonly clock: SimClock;
  private readonly plasticityEngine: PlasticityEngine;
  private populations: ReadonlyMap<number, Population> = new Map();
  private projections: ReadonlyMap<number, Projection> = new Map();
  private eventBus: EventBus | null = null;

  /** External current injection — set by encoders before each step */
  private readonly externalCurrents: Map<number, Float64Array> = new Map();

  /** Last decoded motor output — updated after each step for environment consumption */
  private lastMotorOutput: PopulationSpikeVector = {
    rates: new Float32Array(0),
    windowMs: 30,
    timestamp: 0,
  };

  constructor(config: SimulatorConfig = {}) {
    this.clock = new SimClock(config.dt ?? 0.1);
    this.plasticityEngine = new PlasticityEngine(config.plasticityConfig);
  }

  /** Load a built network into the simulator */
  loadNetwork(
    populations: ReadonlyMap<number, Population>,
    projections: ReadonlyMap<number, Projection>
  ): void {
    this.populations = populations;
    this.projections = projections;

    // Initialize external current arrays for each population
    for (const [id, pop] of populations) {
      this.externalCurrents.set(id, new Float64Array(pop.config.size));
    }
  }

  /** Attach an observability event bus */
  attachEventBus(bus: EventBus): void {
    this.eventBus = bus;
  }

  /** Set external current injection for a population (from encoders) */
  setExternalCurrents(populationId: number, currents: Float64Array): void {
    this.externalCurrents.set(populationId, currents);
  }

  /** Inject current into specific neurons (used by encoders) */
  injectCurrent(populationId: number, neuronIndices: Uint32Array, currents: Float32Array): void {
    const arr = this.externalCurrents.get(populationId);
    if (!arr) return;
    // Reset to zero first — don't accumulate across steps
    arr.fill(0);
    for (let i = 0; i < neuronIndices.length; i++) {
      const idx = neuronIndices[i] as number;
      arr[idx] = (arr[idx] as number) + (currents[i] as number);
    }
  }

  /**
   * Run one simulation timestep.
   * Returns spike events for all populations.
   */
  step(): SimulationStep {
    this.clock.tick();
    const t = this.clock.t;
    const dt = this.clock.dt;

    // Spikes generated this step: population → local indices
    const spikesByPopulation = new Map<number, number[]>();

    // 1. Deliver queued spikes from delay lines (spikes from previous steps)
    for (const proj of this.projections.values()) {
      const arrivedSpikes = proj.delayLine.dequeue();
      proj.delayLine.advance();

      if (arrivedSpikes.length > 0) {
        const targetPop = this.populations.get(proj.config.targetPopulationId);
        if (targetPop) {
          const model = getModel(targetPop.config.modelId);
          deliverSpikes(arrivedSpikes, targetPop, proj.store, proj.config.receptorType, model.stateVectorSize);
        }
      }
    }

    // 2. Step neuron dynamics and collect spikes
    for (const [popId, population] of this.populations) {
      const model = getModel(population.config.modelId);
      const currents = this.externalCurrents.get(popId) ?? new Float64Array(population.config.size);

      const result = stepPopulation(population, model, currents, dt, t);

      if (result.spikeIndices.length > 0) {
        spikesByPopulation.set(popId, result.spikeIndices);

        // Emit spike events to observability bus
        if (this.eventBus) {
          for (const neuronIndex of result.spikeIndices) {
            this.eventBus.emit({ type: 'spike', populationId: popId, neuronIndex, t });
          }
        }
      } else {
        spikesByPopulation.set(popId, []);
      }
    }

    // 3. Enqueue new spikes into delay lines (for delivery in future steps)
    for (const proj of this.projections.values()) {
      const srcSpikes = spikesByPopulation.get(proj.config.sourcePopulationId) ?? [];
      const sourceConfig = this.populations.get(proj.config.sourcePopulationId)?.config;
      if (!sourceConfig) continue;

      for (const srcIdx of srcSpikes) {
        const outgoing = proj.store.getOutgoing(srcIdx);
        for (const syn of outgoing) {
          const delaySteps = Math.max(1, Math.round(syn.delay / dt));
          proj.delayLine.enqueue(srcIdx, delaySteps);
        }
      }
    }

    // 4. Apply plasticity (STDP + homeostasis)
    const preSpikesByProj = new Map<number, readonly number[]>();
    const postSpikesByProj = new Map<number, readonly number[]>();

    for (const [projId, proj] of this.projections) {
      preSpikesByProj.set(projId, spikesByPopulation.get(proj.config.sourcePopulationId) ?? []);
      postSpikesByProj.set(projId, spikesByPopulation.get(proj.config.targetPopulationId) ?? []);
    }

    const plasticityResult = this.plasticityEngine.step(
      this.projections,
      this.populations,
      preSpikesByProj,
      postSpikesByProj,
      dt,
      t
    );

    // Emit homeostatic update events
    if (plasticityResult.homeostaticStates && this.eventBus) {
      for (const [popId, state] of plasticityResult.homeostaticStates) {
        const meanRate = Array.from(state.currentRates).reduce((a, b) => a + b, 0) / state.currentRates.length;
        const meanScaling = Array.from(state.scalingFactors).reduce((a, b) => a + b, 0) / state.scalingFactors.length;
        this.eventBus.emit({
          type: 'homeostasis_update',
          populationId: popId,
          meanScalingFactor: meanScaling,
          meanRate,
          targetRate: state.targetRate,
          t,
        });
      }
    }

    // 5. Reset external currents for next step
    for (const arr of this.externalCurrents.values()) {
      arr.fill(0);
    }

    const totalSpikes = Array.from(spikesByPopulation.values())
      .reduce((sum, spikes) => sum + spikes.length, 0);

    return {
      t,
      step: this.clock.step,
      totalSpikes,
      spikesByPopulation,
    };
  }

  /**
   * Run for a specified duration.
   * @param durationMs  Simulation time to run in ms
   * @param onStep      Optional callback called after each step
   */
  async run(
    durationMs: number,
    onStep?: (result: SimulationStep) => void
  ): Promise<{ totalSteps: number; totalSpikes: number }> {
    const stepsToRun = Math.ceil(durationMs / this.clock.dt);
    let totalSpikes = 0;

    for (let i = 0; i < stepsToRun; i++) {
      const result = this.step();
      totalSpikes += result.totalSpikes;
      onStep?.(result);
    }

    return { totalSteps: stepsToRun, totalSpikes };
  }

  get currentTime(): number {
    return this.clock.t;
  }

  get currentStep(): number {
    return this.clock.step;
  }

  reset(): void {
    this.clock.reset();
    for (const arr of this.externalCurrents.values()) {
      arr.fill(0);
    }
  }
}
