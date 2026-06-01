/**
 * Synaptic conductance updates — the mechanism by which spikes become currents.
 *
 * V1: AMPA and GABA-A (single-exponential conductances).
 * V2: NMDA (voltage-dependent, dual-exponential) and GABA-B.
 *
 * Conductances are accumulated into the target neuron's state vector.
 * The actual current I = g(E_rev - V) is computed inside the neuron model step().
 *
 * Why conductance-based instead of current-based?
 *   Current-based: I_syn = w × spike  (ignores voltage — always same current)
 *   Conductance-based: g increases, I = g(E_rev - V) depends on V
 * Conductance-based synapses are essential for:
 *   - NMDA-like voltage dependence (coincidence detection)
 *   - Inhibitory synapses that shunt vs. hyperpolarize depending on V
 *   - Biologically plausible E/I balance
 */

import type { ReceptorType, Population } from '@snn/types';
import { ADEX_STATE, LIF_STATE, MODEL_IDS } from '@snn/types';

/** Conductance state vector index for a given receptor type and model */
function getConductanceIndex(receptorType: ReceptorType, modelId: number): number {
  if (modelId === MODEL_IDS.ADEX) {
    switch (receptorType) {
      case 'AMPA':
      case 'NMDA':
        return ADEX_STATE.GE;
      case 'GABA_A':
      case 'GABA_B':
        return ADEX_STATE.GI;
    }
  } else if (modelId === MODEL_IDS.LIF) {
    switch (receptorType) {
      case 'AMPA':
      case 'NMDA':
        return LIF_STATE.GE;
      case 'GABA_A':
      case 'GABA_B':
        return LIF_STATE.GI;
    }
  }
  return 0;
}

/**
 * Apply a spike to a target neuron's conductance state.
 *
 * This increments the appropriate conductance by the synaptic weight.
 * The conductance then decays naturally in the next neuron step (exponential decay).
 *
 * @param population    Target population (state is modified in-place)
 * @param targetIndex   Local index of the target neuron
 * @param weight        Synaptic weight in nS
 * @param receptorType  Which receptor to activate
 * @param stateSize     State vector size for this model (SoA layout indexing)
 */
export function applySpikeToTarget(
  population: Population,
  targetIndex: number,
  weight: number,
  receptorType: ReceptorType,
  stateSize: number
): void {
  const n = population.config.size;
  const modelId = population.config.modelId;
  const gIdx = getConductanceIndex(receptorType, modelId);

  // SoA indexing: state[field * N + neuronIndex]
  const stateIdx = gIdx * n + targetIndex;
  population.state[stateIdx] = (population.state[stateIdx] as number) + weight;
}

/**
 * Deliver all spikes from the delay line to their target population.
 * Called each timestep after dequeuing from the projection's delay line.
 *
 * @param spikeIndices   Source neuron indices that arrived (from delay line)
 * @param target         Target population (conductances modified in-place)
 * @param store          Connectivity store for this projection
 * @param receptorType   Receptor type for this projection
 * @param stateSize      Target neuron state vector size
 */
export function deliverSpikes(
  spikeIndices: readonly number[],
  target: Population,
  store: import('@snn/types').ConnectivityStore,
  receptorType: ReceptorType,
  stateSize: number
): void {
  for (const srcIdx of spikeIndices) {
    const outgoing = store.getOutgoing(srcIdx);
    for (const syn of outgoing) {
      applySpikeToTarget(target, syn.targetIndex, syn.weight, receptorType, stateSize);
    }
  }
}
