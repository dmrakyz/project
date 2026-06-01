/**
 * Runtime validation for all configuration objects.
 *
 * The validator is strict — it throws on any violation rather than silently
 * correcting values. Researchers need to know when their configurations are
 * outside biologically plausible ranges.
 */

import type { PopulationConfig, ProjectionConfig, STDPConfig, HomeostaticConfig } from '@snn/types';
import { V1_RECEPTOR_TYPES } from '@snn/types';
import {
  MAX_WITHIN_POPULATION_CONNECTIVITY,
  MAX_BETWEEN_POPULATION_CONNECTIVITY,
  DEFAULT_DT,
} from './defaults';

export interface ValidationResult {
  readonly valid: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
}

function ok(): ValidationResult {
  return { valid: true, errors: [], warnings: [] };
}

function fail(errors: string[], warnings: string[] = []): ValidationResult {
  return { valid: false, errors, warnings };
}

export function assertValid(result: ValidationResult, context: string): void {
  if (!result.valid) {
    throw new Error(
      `Configuration validation failed in ${context}:\n` +
      result.errors.map(e => `  ERROR: ${e}`).join('\n')
    );
  }
  for (const w of result.warnings) {
    console.warn(`[SNN Config Warning] ${context}: ${w}`);
  }
}

export function validatePopulation(config: PopulationConfig): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (config.size < 1) errors.push(`Population size must be ≥ 1, got ${config.size}`);
  if (config.size > 1_000_000) warnings.push(`Very large population (${config.size} neurons) — memory budget should be checked`);
  if (!config.name) errors.push('Population name is required');

  return errors.length > 0 ? fail(errors, warnings) : { valid: true, errors: [], warnings };
}

export function validateProjection(
  config: ProjectionConfig,
  sourceSize: number,
  targetSize: number
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!config.name) errors.push('Projection name is required');

  if (!V1_RECEPTOR_TYPES.has(config.receptorType)) {
    warnings.push(`Receptor type ${config.receptorType} is not in V1 scope — will fall back to AMPA`);
  }

  // Enforce maximum connectivity density
  if (config.connectivity.type === 'all_to_all') {
    const isSelf = config.sourcePopulationId === config.targetPopulationId;
    const limit = isSelf ? MAX_WITHIN_POPULATION_CONNECTIVITY : MAX_BETWEEN_POPULATION_CONNECTIVITY;
    errors.push(
      `all_to_all connectivity gives 100% density, exceeding the ${(limit * 100).toFixed(0)}% limit. ` +
      `Use random connectivity with probability ≤ ${limit} for large populations.`
    );
  }

  if (config.connectivity.type === 'random') {
    const isSelf = config.sourcePopulationId === config.targetPopulationId;
    const limit = isSelf ? MAX_WITHIN_POPULATION_CONNECTIVITY : MAX_BETWEEN_POPULATION_CONNECTIVITY;
    if (config.connectivity.probability > limit) {
      errors.push(
        `Random connectivity probability ${config.connectivity.probability} exceeds ` +
        `${isSelf ? 'within' : 'between'}-population limit of ${limit}. ` +
        `This would exhaust memory for large populations.`
      );
    }
  }

  // Validate delays — minimum one timestep
  if (config.delays.type === 'constant' && config.delays.value < DEFAULT_DT) {
    errors.push(
      `Delay ${config.delays.value}ms is less than minimum timestep ${DEFAULT_DT}ms. ` +
      `Zero-delay spike delivery violates the architecture. Minimum delay: ${DEFAULT_DT}ms.`
    );
  }

  if (config.delays.type === 'uniform' && config.delays.min < DEFAULT_DT) {
    errors.push(
      `Minimum delay ${config.delays.min}ms is less than minimum timestep ${DEFAULT_DT}ms.`
    );
  }

  // Validate weight distribution
  if (config.weights.type === 'normal' && config.weights.mean < 0) {
    warnings.push('Negative mean weight — this creates inhibitory synapses via an excitatory receptor type');
  }

  // Warn if no plasticity is set on a core projection
  if (config.plasticityRuleId === null && !config.plasticityRuleNote) {
    warnings.push(
      `Projection "${config.name}" has no plasticity rule and no plasticityRuleNote explaining why. ` +
      `Fixed connectivity should be a deliberate choice, not a default.`
    );
  }

  return errors.length > 0 ? fail(errors, warnings) : { valid: true, errors: [], warnings };
}

export function validateSTDPConfig(config: STDPConfig): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (config.aPlus <= 0) errors.push('aPlus must be > 0');
  if (config.aMinus <= 0) errors.push('aMinus must be > 0');
  if (config.tauPlus <= 0) errors.push('tauPlus must be > 0');
  if (config.tauMinus <= 0) errors.push('tauMinus must be > 0');
  if (config.wMin < 0) errors.push('wMin must be ≥ 0');
  if (config.wMax <= config.wMin) errors.push('wMax must be > wMin');

  if (config.aMinus > config.aPlus * 1.5) {
    warnings.push(
      `aMinus (${config.aMinus}) is much larger than aPlus (${config.aPlus}). ` +
      `Strong LTD bias will cause network-wide depression. Typical ratio aMinus/aPlus ≈ 1.0–1.1.`
    );
  }

  return errors.length > 0 ? fail(errors, warnings) : { valid: true, errors: [], warnings };
}

export function validateHomeostaticConfig(config: HomeostaticConfig): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (config.targetRate <= 0) errors.push('targetRate must be > 0 Hz');
  if (config.targetRate > 100) warnings.push(`Very high target rate ${config.targetRate} Hz — consider 1–20 Hz for cortical networks`);
  if (config.updateIntervalSteps < 1000) {
    warnings.push(
      `Homeostatic update interval ${config.updateIntervalSteps} steps is short. ` +
      `Homeostasis timescale must be ≥100× STDP timescale. ` +
      `At dt=0.1ms, STDP τ≈20ms = 200 steps → homeostasis minimum 20000 steps.`
    );
  }
  if (config.scalingStrength > 0.01) {
    warnings.push(`Synaptic scaling strength ${config.scalingStrength} may be too aggressive — typical: 0.0001–0.001`);
  }

  return errors.length > 0 ? fail(errors, warnings) : { valid: true, errors: [], warnings };
}

export function validateSimulationConfig(dt: number, maxNeurons: number): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (dt <= 0) errors.push('dt must be > 0 ms');
  if (dt > 1.0) warnings.push(`Large timestep dt=${dt}ms may miss spike dynamics. Recommended: 0.1ms`);
  if (dt < 0.01) warnings.push(`Very small timestep dt=${dt}ms — simulation will be slow`);

  if (maxNeurons > 100_000) {
    const stateBytes = maxNeurons * 4 * 8;  // 4 state vars, 8 bytes each (f64)
    const paramBytes = maxNeurons * 13 * 8; // 13 AdEx params
    const totalMB = (stateBytes + paramBytes) / (1024 * 1024);
    warnings.push(`${maxNeurons} neurons requires ~${totalMB.toFixed(0)}MB for state/params alone — check memory budget`);
  }

  return errors.length > 0 ? fail(errors, warnings) : { valid: true, errors: [], warnings };
}
