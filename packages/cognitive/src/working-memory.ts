/**
 * Working memory circuit template — E/I balanced persistent activity.
 *
 * This is a CONFIGURATION BLUEPRINT, not a "working memory module."
 * The actual persistent activity (or lack thereof) emerges from network dynamics.
 * This template provides the structural scaffolding and analysis tools.
 *
 * The configuration targets the attractor dynamics known to support working memory:
 *   - Strong recurrent excitatory connections (30–40% within-population)
 *   - Inhibitory interneuron population coupled to maintain E/I balance
 *   - Adaptation parameters that allow sustained but not explosive activity
 *
 * Whether this configuration actually produces stable persistent activity
 * depends on the specific AdEx parameters and projection weights. The
 * analysis tools here measure whether the desired property has emerged.
 *
 * Scientific honesty: working memory in SNNs is a hard open problem.
 * This template increases the probability of observing persistent activity
 * but does not guarantee it.
 */

import type { PopulationConfig, ProjectionConfig } from '@snn/types';
import { MODEL_IDS } from '@snn/types';
import type { WorkingMemoryConfig } from '@snn/types';

export function createWorkingMemoryCircuit(
  config: WorkingMemoryConfig,
  namePrefix: string = 'wm'
): { populationConfigs: PopulationConfig[]; getProjectionConfigs: (idMap: Map<string, number>) => ProjectionConfig[] } {
  const excName = `${namePrefix}_exc`;
  const inhName = `${namePrefix}_inh`;

  const populationConfigs: PopulationConfig[] = [
    {
      name: excName,
      size: config.excSize,
      modelId: MODEL_IDS.ADEX,
      role: 'excitatory',
    },
    {
      name: inhName,
      size: config.inhSize,
      modelId: MODEL_IDS.ADEX,
      role: 'inhibitory',
    },
  ];

  function getProjectionConfigs(idMap: Map<string, number>): ProjectionConfig[] {
    const excId = idMap.get(excName) ?? 0;
    const inhId = idMap.get(inhName) ?? 1;

    return [
      // Recurrent excitation (self-sustaining activity)
      {
        name: `${namePrefix}_ee`,
        sourcePopulationId: excId,
        targetPopulationId: excId,
        receptorType: 'AMPA',
        connectivity: { type: 'random', probability: config.recurrentExcProbability },
        weights: { type: 'lognormal', mu: Math.log(config.recurrentExcWeight) - 0.25, sigma: 0.5 },
        delays: { type: 'uniform', min: 0.5, max: 2.0 },
        plasticityRuleId: 'stdp-nearest-neighbor',
      },
      // E → I (excite the interneurons)
      {
        name: `${namePrefix}_ei`,
        sourcePopulationId: excId,
        targetPopulationId: inhId,
        receptorType: 'AMPA',
        connectivity: { type: 'random', probability: config.eiProbability },
        weights: { type: 'constant', value: 0.5 },
        delays: { type: 'constant', value: 0.5 },
        plasticityRuleId: null,
        plasticityRuleNote: 'E→I interneuron projection is fixed to maintain stable E/I coupling',
      },
      // I → E (inhibit the excitatory population — prevents runaway)
      {
        name: `${namePrefix}_ie`,
        sourcePopulationId: inhId,
        targetPopulationId: excId,
        receptorType: 'GABA_A',
        connectivity: { type: 'random', probability: config.ieProbability },
        weights: { type: 'constant', value: config.inhibitoryWeight },
        delays: { type: 'constant', value: 0.5 },
        plasticityRuleId: null,
        plasticityRuleNote: 'I→E feedback inhibition is fixed to maintain E/I balance',
      },
    ];
  }

  return { populationConfigs, getProjectionConfigs };
}

export const DEFAULT_WORKING_MEMORY_CONFIG: WorkingMemoryConfig = {
  excSize: 200,
  inhSize: 50,
  recurrentExcProbability: 0.08,   // 8% — within-population limit
  eiProbability: 0.05,
  ieProbability: 0.05,
  recurrentExcWeight: 2.0,         // nS — moderate to allow self-sustaining but not explosive
  inhibitoryWeight: 4.0,           // nS — stronger inhibition to prevent runaway
};
