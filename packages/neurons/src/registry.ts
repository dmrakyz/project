/**
 * Neuron model registry — resolves model IDs to implementations.
 *
 * Neuron models must be registered before use. The registry is used by:
 *   - The network builder (to initialize populations)
 *   - The WASM bridge (to verify TypeScript/WASM model correspondence)
 *   - Tests (to iterate all registered models for property verification)
 *
 * Do NOT inherit models from each other — LIF and AdEx are independent
 * implementations of the NeuronModel interface, not a hierarchy.
 */

import type { NeuronModel, ModelId } from '@snn/types';
import { AdExModel } from './adex';
import { LIFModel } from './lif';

const _registry = new Map<ModelId, NeuronModel>();

export function registerModel(model: NeuronModel): void {
  if (_registry.has(model.modelId)) {
    throw new Error(`Model ID ${model.modelId} (${model.name}) is already registered`);
  }
  _registry.set(model.modelId, model);
}

export function getModel(modelId: ModelId): NeuronModel {
  const model = _registry.get(modelId);
  if (!model) {
    throw new Error(
      `No neuron model registered for ID ${modelId}. ` +
      `Registered models: [${[..._registry.keys()].join(', ')}]`
    );
  }
  return model;
}

export function hasModel(modelId: ModelId): boolean {
  return _registry.has(modelId);
}

export function listModels(): NeuronModel[] {
  return [..._registry.values()];
}

// Register V1 models immediately
registerModel(new AdExModel());
registerModel(new LIFModel());
