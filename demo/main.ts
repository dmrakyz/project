/**
 * SNN Browser Demo — uses the actual platform packages.
 *
 * Bypasses NetworkBuilder (and its connectivity validator) so we can wire
 * a real E-I circuit with biologically-realistic probabilities. The validator
 * limits are for large cortical-region connectivity; local E-I circuits need
 * 20-30% within-region connectivity to maintain E-I balance.
 *
 * Simulation loop replicates what Simulator.step() does internally:
 *   1. Dequeue pending spikes from delay lines → deliver to conductances
 *   2. Step membrane dynamics (Exponential Euler)
 *   3. Enqueue new spikes into delay lines
 *   4. Apply STDP
 */

import { AdExModel, stepPopulation, initializePopulationState } from '@snn/neurons';
import { buildCSRStore, CircularDelayLine, deliverSpikes } from '@snn/network';
import { STDPRule } from '@snn/plasticity';
import { SpikeEventBus } from '@snn/observability';
import { SeededRNG } from '@snn/config';
import { defaultAdExParams, fastSpikingAdExParams } from '@snn/config';
import {
  MODEL_IDS,
  ADEX_STATE_SIZE,
  ADEX_PARAMS_SIZE,
  DEFAULT_NEUROMODULATOR_STATE,
} from '@snn/types';
import type { Population, PlasticityContext } from '@snn/types';

// ── Network dimensions ────────────────────────────────────────────────────────
const N_E = 80;
const N_I = 20;
const DT  = 0.1;
const MAX_DELAY_MS = 5.0;
const W_MIN = 0, W_MAX = 6;

// ── STDP rule (E→E only) ─────────────────────────────────────────────────────
const STDP_RULE = new STDPRule({ aPlus: 0.01, aMinus: 0.012, wMin: W_MIN, wMax: W_MAX });

// STDP trace decay constants (exact exponential, τ = 20ms)
const DECAY_PRE  = Math.exp(-DT / 20);
const DECAY_POST = Math.exp(-DT / 20);

// Minimal PlasticityContext stub — STDP ignores neuromodulators/homeostasis in V1
function makeCtx(t: number): PlasticityContext {
  return {
    t,
    dt: DT,
    neuromodulators: DEFAULT_NEUROMODULATOR_STATE,
    homeostatic: {
      targetRate: 5,
      currentRates: new Float32Array(1),
      scalingFactors: new Float32Array(1).fill(1),
      excitabilityOffsets: new Float32Array(1),
    },
  };
}
let CTX = makeCtx(0);

// ── Event bus and observability ───────────────────────────────────────────────
export const bus = new SpikeEventBus();

// Raster: flat [t0, globalId0, t1, globalId1, ...]  (globalId = E: 0..N_E-1, I: N_E..N-1)
export const raster: number[] = [];
export const weightHistory: number[] = [];   // sampled every 100ms sim time
export let simT = 0;
export let totalSpikes = 0;
export let stdpEnabled = true;

// ── Simulation state ──────────────────────────────────────────────────────────
let excPop: Population;
let inhPop: Population;
let rng: SeededRNG;

let storeEE: ReturnType<typeof buildCSRStore>;
let storeEI: ReturnType<typeof buildCSRStore>;
let storeIE: ReturnType<typeof buildCSRStore>;
let storeII: ReturnType<typeof buildCSRStore>;

let dlEE: CircularDelayLine;
let dlEI: CircularDelayLine;
let dlIE: CircularDelayLine;
let dlII: CircularDelayLine;

// Per-neuron STDP traces for E→E projection
let preTraceE: Float32Array;
let postTraceE: Float32Array;

let currentsE: Float64Array;
let currentsI: Float64Array;

const adex = new AdExModel();
let weightSampleCounter = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

function makePop(id: number, size: number, role: 'excitatory' | 'inhibitory'): Population {
  return {
    id,
    config: { name: role, size, modelId: MODEL_IDS.ADEX, role },
    state:       new Float64Array(size * ADEX_STATE_SIZE),
    params:      new Float64Array(size * ADEX_PARAMS_SIZE),
    refractory:  new Float64Array(size),
    spikeCounts: new Float64Array(size),
  };
}

/** Fill SoA params for each neuron from a base template with small Gaussian noise. */
function fillParamsSoA(pop: Population, baseParams: Float64Array): void {
  const n = pop.config.size;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < ADEX_PARAMS_SIZE; k++) {
      const base = baseParams[k] as number;
      // ±5% coefficient of variation (matches NetworkBuilder)
      const noise = 1 + rng.normal(0, 0.05);
      pop.params[k * n + i] = base * noise;
    }
  }
}

// ── Initialization ────────────────────────────────────────────────────────────

export function init(seed = 42): void {
  rng = new SeededRNG(seed);
  simT = 0;
  totalSpikes = 0;
  raster.length = 0;

  // Populations
  excPop = makePop(0, N_E, 'excitatory');
  inhPop = makePop(1, N_I, 'inhibitory');

  fillParamsSoA(excPop, defaultAdExParams());
  fillParamsSoA(inhPop, fastSpikingAdExParams());

  initializePopulationState(excPop, adex);
  initializePopulationState(inhPop, adex);

  currentsE = new Float64Array(N_E);
  currentsI = new Float64Array(N_I);

  // Connectivity — call buildCSRStore directly (no validator, real E-I probabilities)
  storeEE = buildCSRStore(N_E, N_E,
    { type: 'random', probability: 0.10 },
    { type: 'lognormal', mu: Math.log(1.2), sigma: 0.3 },
    { type: 'uniform', min: 0.5, max: MAX_DELAY_MS }, 'AMPA', rng);

  storeEI = buildCSRStore(N_E, N_I,
    { type: 'random', probability: 0.12 },
    { type: 'lognormal', mu: Math.log(0.8), sigma: 0.3 },
    { type: 'uniform', min: 0.5, max: MAX_DELAY_MS }, 'AMPA', rng);

  storeIE = buildCSRStore(N_I, N_E,
    { type: 'random', probability: 0.35 },
    { type: 'lognormal', mu: Math.log(2.5), sigma: 0.3 },
    { type: 'uniform', min: 0.5, max: MAX_DELAY_MS }, 'GABA_A', rng);

  storeII = buildCSRStore(N_I, N_I,
    { type: 'random', probability: 0.15 },
    { type: 'lognormal', mu: Math.log(1.0), sigma: 0.3 },
    { type: 'uniform', min: 0.5, max: MAX_DELAY_MS }, 'GABA_A', rng);

  dlEE = new CircularDelayLine(MAX_DELAY_MS, DT);
  dlEI = new CircularDelayLine(MAX_DELAY_MS, DT);
  dlIE = new CircularDelayLine(MAX_DELAY_MS, DT);
  dlII = new CircularDelayLine(MAX_DELAY_MS, DT);

  preTraceE  = new Float32Array(N_E);
  postTraceE = new Float32Array(N_E);
  weightHistory.length = 0;
  weightSampleCounter  = 0;
}

// ── Simulation step ───────────────────────────────────────────────────────────

export function step(driveMode: 'tonic' | 'burst' | 'off'): void {
  simT += DT;
  CTX = makeCtx(simT);

  // 1. Deliver spikes that have arrived after their axonal delay
  deliverSpikes(dlEE.dequeue(), excPop, storeEE, 'AMPA',   ADEX_STATE_SIZE);
  deliverSpikes(dlEI.dequeue(), inhPop, storeEI, 'AMPA',   ADEX_STATE_SIZE);
  deliverSpikes(dlIE.dequeue(), excPop, storeIE, 'GABA_A', ADEX_STATE_SIZE);
  deliverSpikes(dlII.dequeue(), inhPop, storeII, 'GABA_A', ADEX_STATE_SIZE);

  // 2. External drive to E neurons only
  // Burst: 25ms ON / 175ms OFF — long enough to see at 10× speed
  const Iext = driveMode === 'tonic'  ? 250
             : driveMode === 'burst'  ? ((simT % 200) < 25 ? 700 : 0)
             : 0;
  currentIext = Iext;
  currentsE.fill(Iext);
  currentsI.fill(0);

  // 3. Membrane dynamics (Exponential Euler via AdExModel)
  const { spikeIndices: spkE } = stepPopulation(excPop, adex, currentsE, DT, simT);
  const { spikeIndices: spkI } = stepPopulation(inhPop, adex, currentsI, DT, simT);

  // 4. Enqueue new spikes into delay lines, record in raster
  for (const idx of spkE) {
    totalSpikes++;
    raster.push(simT, idx);
    bus.emit({ type: 'spike', populationId: 0, neuronIndex: idx, t: simT });
    for (const syn of storeEE.getOutgoing(idx))
      dlEE.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
    for (const syn of storeEI.getOutgoing(idx))
      dlEI.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
  }

  for (const idx of spkI) {
    totalSpikes++;
    raster.push(simT, N_E + idx);
    bus.emit({ type: 'spike', populationId: 1, neuronIndex: idx, t: simT });
    for (const syn of storeIE.getOutgoing(idx))
      dlIE.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
    for (const syn of storeII.getOutgoing(idx))
      dlII.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
  }

  // 5. STDP on E→E: decay traces, then apply LTD (pre fires) and LTP (post fires)
  if (stdpEnabled) {
    for (let i = 0; i < N_E; i++) {
      preTraceE[i]  = (preTraceE[i]  as number) * DECAY_PRE;
      postTraceE[i] = (postTraceE[i] as number) * DECAY_POST;
    }

    for (const pre of spkE) {
      for (const syn of storeEE.getOutgoing(pre)) {
        const upd = STDP_RULE.onPreSpike(
          syn.index, preTraceE[pre] as number, postTraceE[syn.targetIndex] as number,
          0, syn.weight, CTX
        );
        if (upd.delta !== 0)
          storeEE.setWeight(syn.index, Math.max(W_MIN, Math.min(W_MAX, syn.weight + upd.delta)));
      }
      preTraceE[pre] = (preTraceE[pre] as number) + 1;
    }

    for (const post of spkE) {
      for (const syn of storeEE.getIncoming(post)) {
        const upd = STDP_RULE.onPostSpike(
          syn.index, preTraceE[syn.sourceIndex] as number, postTraceE[post] as number,
          0, syn.weight, CTX
        );
        if (upd.delta !== 0)
          storeEE.setWeight(syn.index, Math.max(W_MIN, Math.min(W_MAX, syn.weight + upd.delta)));
      }
      postTraceE[post] = (postTraceE[post] as number) + 1;
    }
  }

  // Sample E→E weight every 100ms of sim time
  if (++weightSampleCounter % 1000 === 0) {
    weightHistory.push(meanEEWeight());
    if (weightHistory.length > 600) weightHistory.shift();
  }

  // 6. Advance all delay lines
  dlEE.advance(); dlEI.advance(); dlIE.advance(); dlII.advance();

  // 7. Prune raster entries older than 600ms (lazy, when buffer gets large)
  if (raster.length > 20000) {
    const tCut = simT - 600;
    let start = 0;
    while (start < raster.length && (raster[start] as number) < tCut) start += 2;
    if (start > 0) raster.splice(0, start);
  }
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

export function meanEEWeight(): number {
  const snap = storeEE.getWeightsSnapshot();
  if (!snap.length) return 0;
  let s = 0;
  for (let i = 0; i < snap.length; i++) s += snap[i] as number;
  return s / snap.length;
}

export function popRateHz(windowMs = 200): number {
  const tMin = simT - windowMs;
  let n = 0;
  for (let k = 0; k < raster.length; k += 2)
    if ((raster[k] as number) >= tMin) n++;
  return (n / (N_E + N_I)) / (windowMs / 1000);
}

export const N_TOTAL = N_E + N_I;
export const N_EXC   = N_E;

/** Current drive level in pA — used by renderer to show a drive indicator */
export let currentIext = 0;

export function setStdpEnabled(v: boolean) { stdpEnabled = v; }
