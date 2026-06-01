/**
 * Plasticity system types — the second most load-bearing interface set.
 * Must be stable before V2. Wrong abstractions here require rebuilding the
 * entire plasticity subsystem when adding neuromodulators.
 *
 * Design contract:
 * - PlasticityRule never modifies network topology (topology changes are structural plasticity, V3)
 * - PlasticityRule reads from PlasticityContext but does not own its data
 * - WeightUpdate is returned, not applied directly — the caller applies bounds
 */

/** Neuromodulator concentrations — all in normalized [0, 1] range */
export interface NeuromodulatorState {
  /** Dopamine: reward/punishment signal — gates eligibility traces (V2) */
  dopamine: number;
  /** Acetylcholine: attention/arousal — modulates E/I balance (V2) */
  acetylcholine: number;
  /** Serotonin: mood/timing — modulates plasticity thresholds (V2) */
  serotonin: number;
  /** Norepinephrine: arousal/environmental volatility — affects learning rate (V2) */
  norepinephrine: number;
}

/** Default neuromodulator state for V1 (no neuromodulatory system active) */
export const DEFAULT_NEUROMODULATOR_STATE: Readonly<NeuromodulatorState> = {
  dopamine: 0.0,
  acetylcholine: 0.5,
  serotonin: 0.5,
  norepinephrine: 0.5,
};

/** Homeostatic state for a population — updated on slow timescale */
export interface HomeostaticState {
  /** Target firing rate in Hz */
  targetRate: number;
  /** Current estimated firing rate per neuron in Hz */
  currentRates: Float32Array;
  /** Scaling factors applied to incoming synaptic weights per neuron */
  scalingFactors: Float32Array;
  /** V_T offset applied per neuron for intrinsic excitability (mV) */
  excitabilityOffsets: Float32Array;
}

/** Context passed to every plasticity rule on each event */
export interface PlasticityContext {
  /** Current simulation time in ms */
  readonly t: number;
  /** Current timestep size in ms */
  readonly dt: number;
  /** Neuromodulator concentrations — V1 uses DEFAULT_NEUROMODULATOR_STATE */
  readonly neuromodulators: Readonly<NeuromodulatorState>;
  /** Homeostatic state for the target population */
  readonly homeostatic: Readonly<HomeostaticState>;
}

/** Result of a plasticity rule computation — applied by the simulation engine */
export interface WeightUpdate {
  /** Change in synaptic weight — positive = potentiation, negative = depression */
  readonly delta: number;
  /** Optional: update to the eligibility trace (used by R-STDP) */
  readonly eligibilityDelta?: number;
}

export const NO_UPDATE: WeightUpdate = { delta: 0 };

/** Trace types that a rule may require from the network */
export type TraceType = 'pre' | 'post' | 'eligibility';

/**
 * A plasticity rule modifies synaptic weights based on spike timing and context.
 *
 * Rules are stateless — all state lives in the Projection (pre/post/eligibility traces).
 * This makes rules serializable and testable without network setup.
 */
export interface PlasticityRule {
  readonly ruleId: string;
  readonly description: string;
  /** Which trace types this rule reads — used to pre-compute only what is needed */
  readonly requiredTraces: readonly TraceType[];

  /**
   * Called when a pre-synaptic neuron fires.
   * @param synapseIndex   Index into the projection's connectivity store
   * @param preTrace       Current pre-synaptic trace value for this synapse
   * @param postTrace      Current post-synaptic trace value for target neuron
   * @param eligibility    Current eligibility trace for this synapse
   * @param currentWeight  Current synaptic weight
   * @param context        Plasticity context (time, neuromodulators, homeostasis)
   */
  onPreSpike(
    synapseIndex: number,
    preTrace: number,
    postTrace: number,
    eligibility: number,
    currentWeight: number,
    context: PlasticityContext
  ): WeightUpdate;

  /**
   * Called when a post-synaptic neuron fires.
   * Receives all synapses targeting that neuron.
   */
  onPostSpike(
    synapseIndex: number,
    preTrace: number,
    postTrace: number,
    eligibility: number,
    currentWeight: number,
    context: PlasticityContext
  ): WeightUpdate;

  /**
   * Called every timestep — for continuous rules (homeostasis, trace decay, BCM).
   * Returning undefined means no weight update needed this step.
   */
  onTimestep?(
    synapseIndex: number,
    preTrace: number,
    postTrace: number,
    eligibility: number,
    currentWeight: number,
    context: PlasticityContext
  ): WeightUpdate | undefined;
}

/** Configuration for STDP learning rule */
export interface STDPConfig {
  /** LTP amplitude — potentiation magnitude */
  readonly aPlus: number;
  /** LTD amplitude — depression magnitude (positive value, sign applied internally) */
  readonly aMinus: number;
  /** LTP time constant in ms */
  readonly tauPlus: number;
  /** LTD time constant in ms */
  readonly tauMinus: number;
  /** Minimum weight clamp */
  readonly wMin: number;
  /** Maximum weight clamp */
  readonly wMax: number;
}

/** Default biologically-motivated STDP parameters */
export const DEFAULT_STDP_CONFIG: Readonly<STDPConfig> = {
  aPlus: 0.01,
  aMinus: 0.0105,
  tauPlus: 20.0,
  tauMinus: 20.0,
  wMin: 0.0,
  wMax: 1.0,
};

/** Configuration for homeostatic plasticity */
export interface HomeostaticConfig {
  /** Target firing rate in Hz */
  readonly targetRate: number;
  /** Synaptic scaling strength (multiplicative adjustment per update) */
  readonly scalingStrength: number;
  /** Intrinsic excitability adjustment strength (mV per update) */
  readonly excitabilityStrength: number;
  /**
   * Update interval in simulation steps — must be much larger than STDP timescale.
   * Default 10000 steps = 1000ms at dt=0.1ms (10x slower than STDP τ ≈ 20ms).
   * Recommended: ≥100× STDP tau in timesteps.
   */
  readonly updateIntervalSteps: number;
  /** Window over which to estimate firing rate in ms */
  readonly rateEstimationWindow: number;
}

export const DEFAULT_HOMEOSTATIC_CONFIG: Readonly<HomeostaticConfig> = {
  targetRate: 5.0,
  scalingStrength: 0.001,
  excitabilityStrength: 0.01,
  updateIntervalSteps: 10000,
  rateEstimationWindow: 1000.0,
};
