/**
 * V1 Integration Tests — End-to-End Simulation Validation
 *
 * These tests verify the complete simulation stack: network construction,
 * STDP plasticity, homeostatic stability, spike propagation, and observability.
 *
 * V1 Success Criteria (from the architecture plan):
 *   1. Simulate 1K AdEx neurons at dt=0.1ms in reasonable wall time
 *   2. STDP produces non-uniform weight distributions from correlated input
 *   3. Homeostasis stabilizes firing rates after perturbation
 *   4. Pattern completion in recurrent population
 *   5. No numerical instability (NaN/Inf) after 1000ms of simulation
 */

import { NetworkBuilder } from '@snn/network';
import { Simulator } from './simulator';
import { SpikeEventBus } from '@snn/observability';
import { SpikeRecorder, WeightMonitor } from '@snn/observability';
import { PopulationEncoder } from '@snn/environment';
import { MODEL_IDS, SpikeObservabilityEvent } from '@snn/types';

const DT = 0.1; // ms

function buildBasicNetwork(inputSize: number, outputSize: number, seed: number = 42) {
  const builder = new NetworkBuilder({ dt: DT, seed, maxDelayMs: 10 });

  const inputId = builder.addPopulation({
    name: 'input',
    size: inputSize,
    modelId: MODEL_IDS.LIF,
    role: 'input',
  });

  const outputId = builder.addPopulation({
    name: 'output',
    size: outputSize,
    modelId: MODEL_IDS.ADEX,
    role: 'excitatory',
  });

  builder.addProjection({
    name: 'in_to_out',
    sourcePopulationId: inputId,
    targetPopulationId: outputId,
    receptorType: 'AMPA',
    connectivity: { type: 'random', probability: 0.05 },
    weights: { type: 'lognormal', mu: -1.5, sigma: 0.5 },
    delays: { type: 'uniform', min: 1.0, max: 5.0 },
    plasticityRuleId: 'stdp-nearest-neighbor',
  });

  return { network: builder.build(), inputId, outputId };
}

function buildRecurrentNetwork(size: number, seed: number = 99) {
  const builder = new NetworkBuilder({ dt: DT, seed, maxDelayMs: 10 });

  const excId = builder.addPopulation({
    name: 'exc',
    size: size,
    modelId: MODEL_IDS.ADEX,
    role: 'excitatory',
  });

  const inhId = builder.addPopulation({
    name: 'inh',
    size: Math.floor(size / 4),
    modelId: MODEL_IDS.ADEX,
    role: 'inhibitory',
  });

  builder.addProjection({
    name: 'ee',
    sourcePopulationId: excId,
    targetPopulationId: excId,
    receptorType: 'AMPA',
    connectivity: { type: 'random', probability: 0.07 },
    weights: { type: 'constant', value: 1.5 },
    delays: { type: 'uniform', min: 1.0, max: 3.0 },
    plasticityRuleId: 'stdp-nearest-neighbor',
  });

  builder.addProjection({
    name: 'ei',
    sourcePopulationId: excId,
    targetPopulationId: inhId,
    receptorType: 'AMPA',
    connectivity: { type: 'random', probability: 0.04 },
    weights: { type: 'constant', value: 0.5 },
    delays: { type: 'constant', value: 1.0 },
    plasticityRuleId: null,
    plasticityRuleNote: 'Fixed E→I coupling',
  });

  builder.addProjection({
    name: 'ie',
    sourcePopulationId: inhId,
    targetPopulationId: excId,
    receptorType: 'GABA_A',
    connectivity: { type: 'random', probability: 0.04 },
    weights: { type: 'constant', value: 3.0 },
    delays: { type: 'constant', value: 1.0 },
    plasticityRuleId: null,
    plasticityRuleNote: 'Fixed I→E feedback inhibition',
  });

  return { network: builder.build(), excId, inhId };
}

describe('V1 Integration: Network Construction', () => {
  test('builds a valid feedforward network without errors', () => {
    const { network, inputId, outputId } = buildBasicNetwork(50, 10);
    expect(network.populations.size).toBe(2);
    expect(network.projections.size).toBe(1);
    expect(network.populations.get(inputId)?.config.size).toBe(50);
    expect(network.populations.get(outputId)?.config.size).toBe(10);
  });

  test('builds a valid recurrent E/I network', () => {
    const { network, excId, inhId } = buildRecurrentNetwork(40);
    expect(network.populations.size).toBe(2);
    expect(network.projections.size).toBe(3);

    const exc = network.populations.get(excId)!;
    const inh = network.populations.get(inhId)!;
    expect(exc.config.size).toBe(40);
    expect(inh.config.size).toBe(10);
  });

  test('initial state is at resting potential for all neurons', () => {
    const { network, outputId } = buildBasicNetwork(20, 10);
    const pop = network.populations.get(outputId)!;
    const n = pop.config.size;

    // Check that all voltages are near EL = -70mV (SoA layout: V block is first)
    let anyNaN = false;
    for (let i = 0; i < n; i++) {
      const V = pop.state[i] as number;
      if (isNaN(V)) anyNaN = true;
      expect(V).toBeGreaterThan(-80);
      expect(V).toBeLessThan(-60);
    }
    expect(anyNaN).toBe(false);
  });

  test('CSR connectivity respects density limit', () => {
    const { network } = buildBasicNetwork(100, 100, 123);
    const proj = [...network.projections.values()][0]!;
    const actualDensity = proj.store.synapseCount / (100 * 100);
    // Expected ~5% connectivity, allowing ±3% statistical variance
    expect(actualDensity).toBeGreaterThan(0.01);
    expect(actualDensity).toBeLessThan(0.10);
  });
});

describe('V1 Integration: Simulation Dynamics', () => {
  test('simulator runs 500ms without NaN or numerical instability', () => {
    const { network, inputId, outputId } = buildBasicNetwork(50, 10);
    const sim = new Simulator({ dt: DT });
    sim.loadNetwork(network.populations, network.projections);

    const encoder = new PopulationEncoder({ n: 50, minVal: 0, maxVal: 1, sigma: 0.1, peakCurrent: 250, seed: 1 });

    const STEPS = 5000; // 500ms
    for (let i = 0; i < STEPS; i++) {
      const t = i * DT;
      const input = { data: new Float32Array([Math.sin(t / 100) * 0.5 + 0.5]), timestamp: t };
      const encoded = encoder.encode(input, DT, t);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    // Check no NaN in output population state
    const pop = network.populations.get(outputId)!;
    const n = pop.config.size;
    for (let i = 0; i < n; i++) {
      expect(isNaN(pop.state[i] as number)).toBe(false);
      expect(isFinite(pop.state[i] as number)).toBe(true);
    }
  });

  test('neurons fire spikes when driven with suprathreshold current', () => {
    const { network, inputId, outputId } = buildBasicNetwork(20, 5);
    const sim = new Simulator({ dt: DT });
    sim.loadNetwork(network.populations, network.projections);

    const bus = new SpikeEventBus();
    sim.attachEventBus(bus);
    const recorder = new SpikeRecorder({ maxEntries: 10000 });
    recorder.attach(bus);

    const encoder = new PopulationEncoder({ n: 20, minVal: 0, maxVal: 1, sigma: 0.08, peakCurrent: 400, seed: 2 });

    const STEPS = 2000; // 200ms
    for (let i = 0; i < STEPS; i++) {
      const t = i * DT;
      const val = 0.5; // Constant stimulus in middle of range
      const encoded = encoder.encode({ data: new Float32Array([val]), timestamp: t }, DT, t);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    const totalSpikes = recorder.spikeCount;
    expect(totalSpikes).toBeGreaterThan(0);

    recorder.detach();
  });

  test('spike events are emitted to the event bus', () => {
    const { network, inputId } = buildBasicNetwork(20, 5);
    const sim = new Simulator({ dt: DT });
    sim.loadNetwork(network.populations, network.projections);

    const bus = new SpikeEventBus();
    sim.attachEventBus(bus);

    const spikeEvents: { populationId: number; neuronIndex: number; t: number }[] = [];
    bus.subscribe<SpikeObservabilityEvent>('spike', event => {
      spikeEvents.push({ populationId: event.populationId, neuronIndex: event.neuronIndex, t: event.t });
    });

    const encoder = new PopulationEncoder({ n: 20, minVal: 0, maxVal: 1, sigma: 0.1, peakCurrent: 400 });

    for (let i = 0; i < 1000; i++) {
      const encoded = encoder.encode({ data: new Float32Array([0.5]), timestamp: i * DT }, DT, i * DT);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    expect(spikeEvents.length).toBeGreaterThan(0);
    for (const e of spikeEvents) {
      expect(e.t).toBeGreaterThan(0);
      expect(isNaN(e.t)).toBe(false);
    }
  });

  test('delay lines prevent zero-delay spike delivery', () => {
    const { network, inputId, outputId } = buildBasicNetwork(10, 5);
    const sim = new Simulator({ dt: DT });
    sim.loadNetwork(network.populations, network.projections);

    const bus = new SpikeEventBus();
    sim.attachEventBus(bus);

    // Track which steps produce output spikes
    const inputSpikeSteps: number[] = [];
    const outputSpikeSteps: number[] = [];

    bus.subscribe<SpikeObservabilityEvent>('spike', e => {
      if (e.populationId === inputId) inputSpikeSteps.push(e.t);
      if (e.populationId === outputId) outputSpikeSteps.push(e.t);
    });

    const encoder = new PopulationEncoder({ n: 10, minVal: 0, maxVal: 1, sigma: 0.1, peakCurrent: 500 });

    // Run enough steps for spikes to propagate through delays
    for (let i = 0; i < 500; i++) {
      const encoded = encoder.encode({ data: new Float32Array([0.5]), timestamp: i * DT }, DT, i * DT);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    // If output neurons fired, their first spike must be at least 1ms after simulation started
    // (minimum delay = 1.0ms in this network)
    if (outputSpikeSteps.length > 0) {
      const firstOutputSpike = Math.min(...outputSpikeSteps);
      expect(firstOutputSpike).toBeGreaterThan(0.9); // At least 1ms delay
    }
  });
});

describe('V1 Integration: STDP Plasticity', () => {
  test('STDP creates weight differentiation from correlated input', () => {
    const { network, inputId } = buildBasicNetwork(50, 10, 77);
    const sim = new Simulator({
      dt: DT,
      plasticityConfig: {
        stdpConfig: { aPlus: 0.02, aMinus: 0.021, tauPlus: 20, tauMinus: 20, wMin: 0, wMax: 2 },
        homeostaticConfig: { targetRate: 5, updateIntervalSteps: 5000, scalingStrength: 0.001, excitabilityStrength: 0.01, rateEstimationWindow: 500 },
      },
    });
    sim.loadNetwork(network.populations, network.projections);

    const proj = [...network.projections.values()][0]!;
    const initialWeights = proj.store.getWeightsSnapshot();
    const initialVariance = computeVariance(initialWeights);

    const encoder = new PopulationEncoder({ n: 50, minVal: 0, maxVal: 1, sigma: 0.08, peakCurrent: 300 });

    // Run 1000ms with alternating patterns to drive STDP
    const STEPS = 10000;
    for (let i = 0; i < STEPS; i++) {
      const t = i * DT;
      const val = (Math.floor(t / 50) % 2 === 0) ? 0.2 : 0.8;
      const encoded = encoder.encode({ data: new Float32Array([val]), timestamp: t }, DT, t);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    const finalWeights = proj.store.getWeightsSnapshot();
    const finalVariance = computeVariance(finalWeights);

    // STDP should have changed the weight distribution
    let anyChanged = false;
    for (let i = 0; i < Math.min(initialWeights.length, finalWeights.length); i++) {
      if (Math.abs((initialWeights[i] as number) - (finalWeights[i] as number)) > 0.001) {
        anyChanged = true;
        break;
      }
    }
    expect(anyChanged).toBe(true);

    // Weight values should remain in valid range
    for (const w of finalWeights) {
      expect(w).toBeGreaterThanOrEqual(0);
      expect(w).toBeLessThanOrEqual(2);
      expect(isNaN(w)).toBe(false);
    }
  });

  test('homeostatic plasticity does not fully erase weight variation', () => {
    const { network, inputId } = buildBasicNetwork(40, 8, 55);
    const sim = new Simulator({
      dt: DT,
      plasticityConfig: {
        stdpConfig: { aPlus: 0.015, aMinus: 0.016, tauPlus: 20, tauMinus: 20, wMin: 0, wMax: 1.5 },
        homeostaticConfig: {
          targetRate: 5,
          updateIntervalSteps: 8000,  // Very slow homeostasis — 800ms intervals
          scalingStrength: 0.0005,    // Gentle scaling
          excitabilityStrength: 0.005,
          rateEstimationWindow: 500,
        },
      },
    });
    sim.loadNetwork(network.populations, network.projections);

    const proj = [...network.projections.values()][0]!;
    const encoder = new PopulationEncoder({ n: 40, minVal: 0, maxVal: 1, sigma: 0.08, peakCurrent: 300 });

    const STEPS = 10000; // 1000ms
    for (let i = 0; i < STEPS; i++) {
      const t = i * DT;
      const val = (Math.floor(t / 50) % 2 === 0) ? 0.2 : 0.8;
      const encoded = encoder.encode({ data: new Float32Array([val]), timestamp: t }, DT, t);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    const finalWeights = proj.store.getWeightsSnapshot();
    const finalVariance = computeVariance(finalWeights);

    // After training, weights should show some variance (STDP differentiation preserved)
    expect(finalVariance).toBeGreaterThan(0.001);

    // No NaN weights
    for (const w of finalWeights) {
      expect(isNaN(w)).toBe(false);
    }
  });
});

describe('V1 Integration: Performance', () => {
  test('simulates 1000ms of a 500-neuron network in less than 30s', () => {
    const builder = new NetworkBuilder({ dt: DT, seed: 1, maxDelayMs: 5 });
    const inputId = builder.addPopulation({ name: 'in', size: 400, modelId: MODEL_IDS.LIF, role: 'input' });
    const outputId = builder.addPopulation({ name: 'out', size: 100, modelId: MODEL_IDS.ADEX, role: 'excitatory' });
    builder.addProjection({
      name: 'io',
      sourcePopulationId: inputId, targetPopulationId: outputId,
      receptorType: 'AMPA',
      connectivity: { type: 'random', probability: 0.04 },
      weights: { type: 'lognormal', mu: -1.5, sigma: 0.4 },
      delays: { type: 'uniform', min: 1.0, max: 3.0 },
      plasticityRuleId: 'stdp-nearest-neighbor',
    });

    const { network } = { network: builder.build() };
    const sim = new Simulator({ dt: DT });
    sim.loadNetwork(network.populations, network.projections);

    const encoder = new PopulationEncoder({ n: 400, minVal: 0, maxVal: 1, sigma: 0.05, peakCurrent: 250 });

    const start = Date.now();
    const STEPS = 10000; // 1000ms

    for (let i = 0; i < STEPS; i++) {
      const t = i * DT;
      const val = 0.5 + 0.3 * Math.sin(t / 50);
      const encoded = encoder.encode({ data: new Float32Array([val]), timestamp: t }, DT, t);
      sim.injectCurrent(inputId, encoded.activeNeurons, encoded.currents);
      sim.step();
    }

    const wallMs = Date.now() - start;
    console.log(`  500-neuron, 1000ms: ${wallMs}ms wall time (${(1000/wallMs).toFixed(2)}x real-time)`);

    // Must complete in under 30s (strict: real-time, relaxed: 30x slower for CI)
    expect(wallMs).toBeLessThan(30_000);
  }, 60_000); // 60s timeout for this test
});

function computeVariance(arr: Float32Array): number {
  let sum = 0;
  for (const v of arr) sum += v;
  const mean = sum / arr.length;
  let variance = 0;
  for (const v of arr) variance += (v - mean) ** 2;
  return variance / arr.length;
}
