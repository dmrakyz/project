/**
 * Experiment runner — configuration-driven experiment specification.
 *
 * An Experiment ties together:
 *   - A network configuration (populations + projections)
 *   - An environment (optional)
 *   - Plasticity settings
 *   - Observability configuration
 *   - Duration and evaluation criteria
 *
 * Experiments are specified declaratively (JSON/TypeScript config), not as
 * imperative training loops. This ensures reproducibility via seeded RNG and
 * makes experimental parameters explicit and auditable.
 *
 * No training loops, no epochs, no loss functions.
 * The experiment simply runs the simulation for a specified time and collects data.
 */

import type { BuiltNetwork } from '@snn/network';
import type { Environment, SpikeEncoder, SpikeDecoder } from '@snn/types';
import { Simulator } from './simulator';
import type { SimulationStep } from './simulator';
import { SpikeEventBus } from '@snn/observability';
import { SpikeRecorder, RateEstimator, WeightMonitor } from '@snn/observability';

export interface ExperimentConfig {
  readonly name: string;
  readonly description?: string;
  /** Total simulation duration in ms */
  readonly durationMs: number;
  readonly dt?: number;
  /** Log interval in steps (how often to print status) */
  readonly logIntervalSteps?: number;
  readonly plasticity?: {
    readonly stdp?: Partial<import('@snn/types').STDPConfig>;
    readonly homeostasis?: Partial<import('@snn/types').HomeostaticConfig>;
  };
  readonly observability?: {
    readonly spikeRecorderMaxEntries?: number;
    readonly weightMonitorIntervalSteps?: number;
  };
}

export interface ExperimentResult {
  readonly name: string;
  readonly totalSteps: number;
  readonly totalSimulatedMs: number;
  readonly totalSpikes: number;
  /** Mean firing rates per population at end of simulation */
  readonly finalRates: ReadonlyMap<number, number>;
  readonly spikeRecorder: SpikeRecorder;
  readonly weightMonitor: WeightMonitor;
  readonly wallTimeMs: number;
}

export class Experiment {
  private readonly config: ExperimentConfig;
  private network: BuiltNetwork | null = null;
  private environment: Environment | null = null;
  private encoder: SpikeEncoder | null = null;
  private decoder: SpikeDecoder | null = null;
  private motorPopulationId: number | null = null;

  constructor(config: ExperimentConfig) {
    this.config = config;
  }

  withNetwork(network: BuiltNetwork): this {
    this.network = network;
    return this;
  }

  withEnvironment(
    env: Environment,
    encoder: SpikeEncoder,
    decoder: SpikeDecoder,
    motorPopulationId: number
  ): this {
    this.environment = env;
    this.encoder = encoder;
    this.decoder = decoder;
    this.motorPopulationId = motorPopulationId;
    return this;
  }

  async run(): Promise<ExperimentResult> {
    if (!this.network) throw new Error('No network loaded — call withNetwork() first');

    const startWall = Date.now();
    const config = this.config;
    const dt = config.dt ?? this.network.dt;

    const simulator = new Simulator({
      dt,
      plasticityConfig: {
        stdpConfig: config.plasticity?.stdp,
        homeostaticConfig: config.plasticity?.homeostasis,
      },
    });

    simulator.loadNetwork(this.network.populations, this.network.projections);

    // Observability setup
    const bus = new SpikeEventBus();
    simulator.attachEventBus(bus);

    const recorder = new SpikeRecorder({
      maxEntries: config.observability?.spikeRecorderMaxEntries ?? 500_000,
    });
    recorder.attach(bus);

    const rateEstimator = new RateEstimator(100);
    bus.subscribe('spike', event => {
      rateEstimator.recordSpikes(event.populationId, [event.neuronIndex], event.t);
    });

    const weightMonitor = new WeightMonitor({
      intervalSteps: config.observability?.weightMonitorIntervalSteps ?? 1000,
    });

    const logInterval = config.logIntervalSteps ?? 10000;
    let totalSpikes = 0;

    // Initialize environment
    if (this.environment) {
      this.environment.reset();
    }

    // Current sensor input for the encoder
    let currentInput = this.environment
      ? this.environment.reset()
      : { data: new Float32Array(0), timestamp: 0 };

    let lastDecodedSpikes: number[] = [];

    const stepsToRun = Math.ceil(config.durationMs / dt);

    for (let i = 0; i < stepsToRun; i++) {
      // Encode sensory input as current injections
      if (this.encoder && this.network.populations.size > 0) {
        const encoded = this.encoder.encode(currentInput, dt, simulator.currentTime);
        const inputPopId = [...this.network.populations.keys()][0]!;
        simulator.injectCurrent(inputPopId, encoded.activeNeurons, encoded.currents);
      }

      const result = simulator.step();
      totalSpikes += result.totalSpikes;

      // Collect motor spikes for decoding
      if (this.motorPopulationId !== null) {
        const motorSpikes = result.spikesByPopulation.get(this.motorPopulationId) ?? [];
        lastDecodedSpikes = [...motorSpikes];
      }

      // Advance environment
      if (this.environment && this.decoder) {
        const motorVector: import('@snn/types').PopulationSpikeVector = {
          rates: this.decoder.decode(lastDecodedSpikes, result.t),
          windowMs: 30,
          timestamp: result.t,
        };
        currentInput = this.environment.step(motorVector, dt);

        if (this.environment.isDone()) {
          this.environment.reset();
        }
      }

      // Weight monitor on slow timescale
      weightMonitor.onTimestep(this.network.projections, result.t);

      // Progress logging
      if (i % logInterval === 0) {
        const progress = ((i / stepsToRun) * 100).toFixed(1);
        const meanRate = totalSpikes / (result.t * (this.network.populations.size || 1));
        console.log(`[${config.name}] ${progress}% | t=${result.t.toFixed(0)}ms | spikes/step=${result.totalSpikes} | mean_rate≈${meanRate.toFixed(1)}Hz`);
      }
    }

    // Compute final firing rates
    const finalRates = new Map<number, number>();
    for (const [popId, pop] of this.network.populations) {
      finalRates.set(popId, rateEstimator.estimateRate(popId, pop.config.size, simulator.currentTime));
    }

    recorder.detach();

    return {
      name: config.name,
      totalSteps: stepsToRun,
      totalSimulatedMs: config.durationMs,
      totalSpikes,
      finalRates,
      spikeRecorder: recorder,
      weightMonitor,
      wallTimeMs: Date.now() - startWall,
    };
  }
}
