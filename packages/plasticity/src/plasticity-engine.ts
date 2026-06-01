/**
 * PlasticityEngine — applies plasticity rules to all projections each timestep.
 *
 * The engine is the event bridge between the network simulation and the plasticity rules:
 *   1. Receives pre- and post-synaptic spike events
 *   2. Decays STDP traces
 *   3. Calls rule.onPreSpike / rule.onPostSpike for affected synapses
 *   4. Applies weight updates (bounded by rule config)
 *   5. Dispatches to homeostatic system on its slower timescale
 *
 * Architectural constraints enforced here:
 *   - Weight updates are returned from rules, NOT applied by rules directly
 *   - Rules receive only their required trace types (declared in requiredTraces)
 *   - Homeostasis runs at a configurable slower interval (never the same rate as STDP)
 */

import type { Projection, Population } from '@snn/types';
import { DEFAULT_STDP_CONFIG } from '@snn/types';
import {
  STDPRule,
  decayTraces,
  incrementPreTrace,
  incrementPostTrace,
} from './stdp';
import { HomeostaticPlasticity } from './homeostasis';

export interface PlasticityEngineConfig {
  readonly stdpConfig?: Partial<import('@snn/types').STDPConfig>;
  readonly homeostaticConfig?: Partial<import('@snn/types').HomeostaticConfig>;
}

export interface PlasticityEngineResult {
  readonly weightUpdatesApplied: number;
  /** Non-null only on homeostatic update steps */
  readonly homeostaticStates?: ReturnType<HomeostaticPlasticity['onTimestep']>;
}

export class PlasticityEngine {
  private readonly stdpRule: STDPRule;
  private readonly homeostasis: HomeostaticPlasticity;
  private readonly tauPre: number;
  private readonly tauPost: number;
  private readonly wMin: number;
  private readonly wMax: number;

  /** Registry of plasticity rules by ruleId — V2 will add R-STDP here */
  private readonly rules: Map<string, import('@snn/types').PlasticityRule> = new Map();

  constructor(config: PlasticityEngineConfig = {}) {
    this.stdpRule = new STDPRule(config.stdpConfig);
    this.homeostasis = new HomeostaticPlasticity(config.homeostaticConfig);

    const stdpCfg = this.stdpRule.getConfig();
    this.tauPre = stdpCfg.tauPlus;
    this.tauPost = stdpCfg.tauMinus;
    this.wMin = stdpCfg.wMin;
    this.wMax = stdpCfg.wMax;

    this.rules.set(this.stdpRule.ruleId, this.stdpRule);
  }

  /**
   * Process all plasticity for one timestep.
   *
   * @param projections       All projections in the network
   * @param populations       All populations (for homeostasis access)
   * @param preSpikesByProj   Map from projectionId → source spike indices this step
   * @param postSpikesByProj  Map from projectionId → target spike indices this step
   * @param dt                Simulation timestep in ms
   * @param t                 Current simulation time in ms
   */
  step(
    projections: ReadonlyMap<number, Projection>,
    populations: ReadonlyMap<number, Population>,
    preSpikesByProj: ReadonlyMap<number, readonly number[]>,
    postSpikesByProj: ReadonlyMap<number, readonly number[]>,
    dt: number,
    t: number
  ): PlasticityEngineResult {
    let weightUpdatesApplied = 0;

    for (const [projId, projection] of projections) {
      if (projection.config.plasticityRuleId === null) continue;

      const rule = this.rules.get(projection.config.plasticityRuleId);
      if (!rule) continue;

      const preSpikes  = preSpikesByProj.get(projId) ?? [];
      const postSpikes = postSpikesByProj.get(projId) ?? [];

      // 1. Decay traces (exact exponential)
      decayTraces(projection.preTrace, projection.postTrace, dt, this.tauPre, this.tauPost);

      // 2. Process pre-synaptic spikes → update pre traces + apply LTD
      for (const srcIdx of preSpikes) {
        const outgoing = projection.store.getOutgoing(srcIdx);
        const outgoingIndices = outgoing.map(s => s.index);
        incrementPreTrace(projection.preTrace, outgoingIndices);

        for (const syn of outgoing) {
          const postTrace = projection.postTrace[syn.targetIndex] as number;
          const update = rule.onPreSpike(
            syn.index,
            projection.preTrace[syn.index] as number,
            postTrace,
            projection.eligibilityTrace[syn.index] as number,
            syn.weight,
            { t, dt, neuromodulators: { dopamine: 0, acetylcholine: 0.5, serotonin: 0.5, norepinephrine: 0.5 },
              homeostatic: { targetRate: 5, currentRates: new Float32Array(1), scalingFactors: new Float32Array(1).fill(1), excitabilityOffsets: new Float32Array(1) } }
          );

          if (update.delta !== 0) {
            const newWeight = Math.max(this.wMin, Math.min(this.wMax, syn.weight + update.delta));
            projection.store.setWeight(syn.index, newWeight);
            weightUpdatesApplied++;
          }
        }
      }

      // 3. Process post-synaptic spikes → update post traces + apply LTP
      for (const tgtIdx of postSpikes) {
        incrementPostTrace(projection.postTrace, tgtIdx);
        const incoming = projection.store.getIncoming(tgtIdx);

        for (const syn of incoming) {
          const preTrace = projection.preTrace[syn.index] as number;
          const update = rule.onPostSpike(
            syn.index,
            preTrace,
            projection.postTrace[tgtIdx] as number,
            projection.eligibilityTrace[syn.index] as number,
            syn.weight,
            { t, dt, neuromodulators: { dopamine: 0, acetylcholine: 0.5, serotonin: 0.5, norepinephrine: 0.5 },
              homeostatic: { targetRate: 5, currentRates: new Float32Array(1), scalingFactors: new Float32Array(1).fill(1), excitabilityOffsets: new Float32Array(1) } }
          );

          if (update.delta !== 0) {
            const newWeight = Math.max(this.wMin, Math.min(this.wMax, syn.weight + update.delta));
            projection.store.setWeight(syn.index, newWeight);
            weightUpdatesApplied++;
          }
        }
      }

      // 4. Homeostasis: record spikes for rate estimation
      const targetPopId = projection.config.targetPopulationId;
      const targetPop = populations.get(targetPopId);
      if (targetPop) {
        this.homeostasis.recordSpikes(targetPopId, postSpikes as number[], targetPop.config.size);
      }
    }

    // 5. Apply homeostasis on its slow timescale
    const homeostaticStates = this.homeostasis.onTimestep(populations, projections, dt);

    return { weightUpdatesApplied, homeostaticStates };
  }
}
