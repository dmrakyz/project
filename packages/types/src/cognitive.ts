/**
 * Cognitive architecture types — circuit templates and analysis interfaces.
 *
 * These are configuration blueprints, not "cognition modules."
 * The actual computation emerges from the network dynamics given these configurations.
 * This package provides templates + analysis tools to verify desired properties emerge.
 */

/** A circuit template: named configuration for a functionally-motivated subnetwork */
export interface CircuitTemplate {
  readonly name: string;
  readonly description: string;
  readonly version: number;

  /**
   * Create the population configurations needed for this circuit.
   * Returns configs that should be passed to the network builder.
   */
  getPopulationConfigs(): import('./network.js').PopulationConfig[];

  /**
   * Create the projection configurations for this circuit.
   * Population IDs in projections are indices into the array returned by getPopulationConfigs().
   * The caller resolves these to actual population IDs after network construction.
   */
  getProjectionConfigs(
    populationIdMap: ReadonlyMap<string, number>
  ): import('./network.js').ProjectionConfig[];
}

/** Configuration for a working memory circuit (E/I balanced persistent activity) */
export interface WorkingMemoryConfig {
  /** Size of excitatory population */
  readonly excSize: number;
  /** Size of inhibitory interneuron population */
  readonly inhSize: number;
  /** Recurrent excitatory connection probability (recommended: 0.3–0.4) */
  readonly recurrentExcProbability: number;
  /** E→I connection probability */
  readonly eiProbability: number;
  /** I→E connection probability */
  readonly ieProbability: number;
  /** Recurrent excitatory weight (nS) */
  readonly recurrentExcWeight: number;
  /** I→E inhibitory weight (nS) */
  readonly inhibitoryWeight: number;
}

/** Analysis result for persistent activity detection */
export interface PersistentActivityAnalysis {
  /** Whether persistent activity was detected after stimulus removal */
  readonly detected: boolean;
  /** Duration of sustained activity in ms (0 if not detected) */
  readonly durationMs: number;
  /** Mean firing rate during sustained period in Hz */
  readonly meanRate: number;
  /** Coefficient of variation of inter-spike intervals (regularity measure) */
  readonly isiCV: number;
}
