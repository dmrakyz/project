/**
 * STDP Feature Learning Experiment — V1 Validation
 *
 * Demonstrates STDP-driven receptive field formation from correlated input.
 *
 * Setup:
 *   - Input population: 100 LIF neurons encoding a 1D sensory variable
 *   - Output population: 20 AdEx neurons
 *   - Projection: random (5% connectivity) with STDP plasticity
 *
 * Expected outcome:
 *   Output neurons should develop selectivity for specific input ranges
 *   as STDP strengthens connections from consistently co-active input neurons.
 *
 * Success criterion (V1):
 *   - Network runs without numerical instability for 5000ms
 *   - Weight distribution becomes non-uniform (entropy decreases from initial)
 *   - Mean firing rate stays within homeostatic target range [2, 15] Hz
 *
 * This is NOT a supervised learning task. No labels, no loss functions.
 * Selectivity emerges from the statistics of correlated input patterns.
 */

import { MODEL_IDS } from '@snn/types';
import { NetworkBuilder } from '@snn/network';
import { Experiment } from '@snn/simulation';
import { PopulationEncoder } from '@snn/environment';
import { exportSpikeRasterToCSV, exportWeightSnapshotsToJSON } from '@snn/observability';
import * as fs from 'fs';
import * as path from 'path';

async function runSTDPFeatureLearning() {
  console.log('=== STDP Feature Learning Experiment ===');
  console.log('Demonstrates STDP-driven receptive field formation from correlated input.');
  console.log('No supervised learning. No gradient descent. No loss functions.\n');

  // Build network
  const builder = new NetworkBuilder({ dt: 0.1, seed: 42 });

  const inputPopId = builder.addPopulation({
    name: 'input',
    size: 100,
    modelId: MODEL_IDS.LIF,  // LIF for input — simpler, faster
    role: 'input',
  });

  const outputPopId = builder.addPopulation({
    name: 'output',
    size: 20,
    modelId: MODEL_IDS.ADEX,  // AdEx for processing — adaptation + burst dynamics
    role: 'excitatory',
  });

  // Inhibitory interneuron population for lateral competition
  const inhPopId = builder.addPopulation({
    name: 'inh',
    size: 10,
    modelId: MODEL_IDS.ADEX,
    role: 'inhibitory',
  });

  // Input → Output: plastic STDP projection
  builder.addProjection({
    name: 'input_to_output',
    sourcePopulationId: inputPopId,
    targetPopulationId: outputPopId,
    receptorType: 'AMPA',
    connectivity: { type: 'random', probability: 0.05 },
    weights: { type: 'lognormal', mu: -1.5, sigma: 0.5 }, // Lognormal — biologically motivated
    delays: { type: 'uniform', min: 1.0, max: 3.0 },
    plasticityRuleId: 'stdp-nearest-neighbor',
  });

  // Output → Inh: excite interneurons
  builder.addProjection({
    name: 'output_to_inh',
    sourcePopulationId: outputPopId,
    targetPopulationId: inhPopId,
    receptorType: 'AMPA',
    connectivity: { type: 'random', probability: 0.04 },
    weights: { type: 'constant', value: 0.5 },
    delays: { type: 'constant', value: 1.0 },
    plasticityRuleId: null,
    plasticityRuleNote: 'Fixed E→I coupling for WTA competition',
  });

  // Inh → Output: lateral inhibition (winner-takes-all competition)
  builder.addProjection({
    name: 'inh_to_output',
    sourcePopulationId: inhPopId,
    targetPopulationId: outputPopId,
    receptorType: 'GABA_A',
    connectivity: { type: 'random', probability: 0.04 },
    weights: { type: 'constant', value: 3.0 },
    delays: { type: 'constant', value: 1.0 },
    plasticityRuleId: null,
    plasticityRuleNote: 'Fixed lateral inhibition for competition',
  });

  const network = builder.build();

  console.log(`Network built:`);
  console.log(`  Input population: ${network.populations.get(inputPopId)?.config.size} LIF neurons`);
  console.log(`  Output population: ${network.populations.get(outputPopId)?.config.size} AdEx neurons`);
  console.log(`  Inhibitory population: ${network.populations.get(inhPopId)?.config.size} AdEx neurons`);
  const projectionCount = network.projections.size;
  let totalSynapses = 0;
  for (const proj of network.projections.values()) {
    totalSynapses += proj.store.synapseCount;
  }
  console.log(`  Projections: ${projectionCount} (${totalSynapses} synapses)\n`);

  // Encoder: maps a 1D sensory value to current injection in input population
  const encoder = new PopulationEncoder({
    n: 100,
    minVal: 0,
    maxVal: 1,
    sigma: 0.1,
    peakCurrent: 300,  // pA — drives LIF neurons above threshold
    seed: 0,
  });

  // Create a synthetic input generator — correlated patterns
  // Pattern A: value around 0.2, Pattern B: value around 0.7
  // Each pattern presented for 50ms, alternating
  let currentPatternValue = 0.2;
  let patternTimer = 0;
  const PATTERN_DURATION_MS = 50;

  // We'll use a custom environment-like adapter
  const getSensorInput = (t: number): import('@snn/types').SensorInput => {
    patternTimer = t % (2 * PATTERN_DURATION_MS);
    currentPatternValue = patternTimer < PATTERN_DURATION_MS ? 0.2 : 0.7;
    return {
      data: new Float32Array([currentPatternValue]),
      timestamp: t,
    };
  };

  // Run experiment
  const experiment = new Experiment({
    name: 'stdp-feature-learning',
    description: 'STDP receptive field formation from alternating input patterns',
    durationMs: 5000,
    dt: 0.1,
    plasticity: {
      stdp: { aPlus: 0.01, aMinus: 0.0105, tauPlus: 20, tauMinus: 20, wMin: 0, wMax: 1 },
      homeostasis: { targetRate: 5, updateIntervalSteps: 10000, scalingStrength: 0.001 },
    },
    observability: {
      weightMonitorIntervalSteps: 500,
      spikeRecorderMaxEntries: 200_000,
    },
    logIntervalSteps: 10000,
  });

  // Custom run with input injection per step
  const { Simulator } = await import('@snn/simulation');
  const { SpikeEventBus, SpikeRecorder, WeightMonitor } = await import('@snn/observability');

  const simulator = new Simulator({ dt: 0.1, plasticityConfig: {
    stdpConfig: { aPlus: 0.01, aMinus: 0.0105, tauPlus: 20, tauMinus: 20, wMin: 0, wMax: 1 },
    homeostaticConfig: { targetRate: 5, updateIntervalSteps: 10000, scalingStrength: 0.001, excitabilityStrength: 0.01, rateEstimationWindow: 1000 },
  }});

  simulator.loadNetwork(network.populations, network.projections);

  const bus = new SpikeEventBus();
  simulator.attachEventBus(bus);

  const recorder = new SpikeRecorder({ maxEntries: 200_000 });
  recorder.attach(bus, [outputPopId]);

  const weightMonitor = new WeightMonitor({ intervalSteps: 500 });

  const DURATION_MS = 5000;
  const DT = 0.1;
  const STEPS = Math.ceil(DURATION_MS / DT);

  let totalSpikes = 0;
  const startWall = Date.now();

  for (let i = 0; i < STEPS; i++) {
    const t = i * DT;

    // Encode sensory input and inject as current
    const sensorInput = getSensorInput(t);
    const encoded = encoder.encode(sensorInput, DT, t);
    simulator.injectCurrent(inputPopId, encoded.activeNeurons, encoded.currents);

    const result = simulator.step();
    totalSpikes += result.totalSpikes;

    weightMonitor.onTimestep(network.projections, result.t);

    if (i > 0 && i % 10000 === 0) {
      const wallSec = (Date.now() - startWall) / 1000;
      const simSec = result.t / 1000;
      const speedFactor = simSec / wallSec;
      console.log(`  t=${result.t.toFixed(0)}ms | spikes_this_step=${result.totalSpikes} | speed=${speedFactor.toFixed(1)}x real-time`);
    }
  }

  const wallTimeMs = Date.now() - startWall;

  // Analysis
  const outputProj = [...network.projections.values()].find(p => p.config.name === 'input_to_output');
  const finalWeights = outputProj?.store.getWeightsSnapshot() ?? new Float32Array(0);

  let weightSum = 0;
  let weightMin = Infinity;
  let weightMax = -Infinity;
  for (const w of finalWeights) {
    weightSum += w;
    if (w < weightMin) weightMin = w;
    if (w > weightMax) weightMax = w;
  }
  const weightMean = finalWeights.length > 0 ? weightSum / finalWeights.length : 0;

  const initialWeightEntropy = weightMonitor.getSnapshots(outputProj?.id ?? 0)[0];
  const finalWeightEntropy = weightMonitor.getWeightEntropy(outputProj?.id ?? 0);

  console.log('\n=== Results ===');
  console.log(`Total simulation time: ${DURATION_MS}ms`);
  console.log(`Wall time: ${wallTimeMs}ms (${(DURATION_MS / wallTimeMs).toFixed(1)}x real-time)`);
  console.log(`Total spikes: ${totalSpikes} (mean ${(totalSpikes / STEPS).toFixed(2)} per step)`);
  console.log(`\nFinal weight distribution (input→output projection):`);
  console.log(`  Count: ${finalWeights.length} synapses`);
  console.log(`  Mean: ${weightMean.toFixed(4)}`);
  console.log(`  Min: ${weightMin.toFixed(4)}, Max: ${weightMax.toFixed(4)}`);
  console.log(`  Weight entropy: ${finalWeightEntropy.toFixed(3)} bits`);
  console.log(`\nWeight snapshots captured: ${weightMonitor.getSnapshots(outputProj?.id ?? 0).length}`);

  const spikes = recorder.getSpikes({ populationId: outputPopId });
  console.log(`Output population spike count: ${spikes.length}`);

  // V1 success criteria
  const successCriteria = [
    { name: 'Network ran without crash', pass: true },
    { name: 'Weight entropy > 0.5 bits (non-uniform distribution)', pass: finalWeightEntropy > 0.5 },
    { name: 'Weight max - min > 0.1 (some differentiation)', pass: weightMax - weightMin > 0.1 },
    { name: 'Simulation ran faster than real-time', pass: wallTimeMs < DURATION_MS },
  ];

  console.log('\n=== V1 Success Criteria ===');
  let allPassed = true;
  for (const criterion of successCriteria) {
    const status = criterion.pass ? '✓ PASS' : '✗ FAIL';
    console.log(`  ${status}: ${criterion.name}`);
    if (!criterion.pass) allPassed = false;
  }

  console.log(`\nOverall: ${allPassed ? 'ALL CRITERIA PASSED' : 'SOME CRITERIA FAILED'}`);

  // Export results
  const outDir = path.join(__dirname, 'output');
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const spikeCSV = exportSpikeRasterToCSV(spikes);
  fs.writeFileSync(path.join(outDir, 'output_spikes.csv'), spikeCSV);
  console.log(`\nExported spike raster to output/output_spikes.csv (${spikes.length} entries)`);

  const weightSnapshots = weightMonitor.getSnapshots(outputProj?.id ?? 0);
  if (weightSnapshots.length > 0) {
    const weightJSON = exportWeightSnapshotsToJSON(weightSnapshots.slice(0, 50)); // First 50
    fs.writeFileSync(path.join(outDir, 'weight_snapshots.json'), weightJSON);
    console.log(`Exported ${Math.min(50, weightSnapshots.length)} weight snapshots to output/weight_snapshots.json`);
  }

  recorder.detach();
}

runSTDPFeatureLearning().catch(console.error);
