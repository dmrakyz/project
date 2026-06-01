/**
 * Observability types — scientific instrumentation, not logging.
 *
 * Every subsystem emits typed events into a common event bus.
 * Researchers subscribe to events by type and population ID.
 * Overhead is zero when no subscribers are attached.
 */

export type ObservabilityEventType =
  | 'spike'
  | 'weight_update'
  | 'homeostasis_update'
  | 'neuromodulator_update'
  | 'population_rate'
  | 'state_snapshot'
  | 'checkpoint';

export interface SpikeObservabilityEvent {
  readonly type: 'spike';
  readonly populationId: number;
  readonly neuronIndex: number;
  readonly t: number;
}

export interface WeightUpdateObservabilityEvent {
  readonly type: 'weight_update';
  readonly projectionId: number;
  readonly synapseIndex: number;
  readonly oldWeight: number;
  readonly newWeight: number;
  readonly t: number;
}

export interface HomeostaticUpdateObservabilityEvent {
  readonly type: 'homeostasis_update';
  readonly populationId: number;
  readonly meanScalingFactor: number;
  readonly meanRate: number;
  readonly targetRate: number;
  readonly t: number;
}

export interface PopulationRateObservabilityEvent {
  readonly type: 'population_rate';
  readonly populationId: number;
  /** Firing rate per neuron in Hz, averaged over estimation window */
  readonly rates: Float32Array;
  readonly windowMs: number;
  readonly t: number;
}

export interface StateSnapshotObservabilityEvent {
  readonly type: 'state_snapshot';
  readonly populationId: number;
  /** Full SoA state matrix snapshot */
  readonly state: Float64Array;
  readonly t: number;
}

export type ObservabilityEvent =
  | SpikeObservabilityEvent
  | WeightUpdateObservabilityEvent
  | HomeostaticUpdateObservabilityEvent
  | PopulationRateObservabilityEvent
  | StateSnapshotObservabilityEvent;

export type ObservabilityEventHandler<T extends ObservabilityEvent = ObservabilityEvent> =
  (event: T) => void;

/** The event bus — the central observability hub */
export interface EventBus {
  /** Subscribe to events of a specific type, optionally filtered by population/projection */
  subscribe<T extends ObservabilityEvent>(
    eventType: T['type'],
    handler: ObservabilityEventHandler<T>,
    filter?: EventFilter
  ): Unsubscribe;

  /** Emit an event — called by simulation subsystems */
  emit(event: ObservabilityEvent): void;

  /** Remove all subscribers */
  clear(): void;

  /** Number of currently active subscriptions */
  readonly subscriptionCount: number;
}

export interface EventFilter {
  readonly populationId?: number;
  readonly projectionId?: number;
}

export type Unsubscribe = () => void;

/** Spike raster entry for export */
export interface SpikeRasterEntry {
  readonly t: number;
  readonly populationId: number;
  readonly neuronIndex: number;
}

/** Weight distribution snapshot for export */
export interface WeightSnapshot {
  readonly t: number;
  readonly projectionId: number;
  readonly weights: Float32Array;
  readonly mean: number;
  readonly std: number;
  readonly min: number;
  readonly max: number;
}
