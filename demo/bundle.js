"use strict";
var SNN = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // demo/main.ts
  var main_exports = {};
  __export(main_exports, {
    N_EXC: () => N_EXC,
    N_TOTAL: () => N_TOTAL,
    bus: () => bus,
    currentIext: () => currentIext,
    init: () => init,
    meanEEWeight: () => meanEEWeight,
    popRateHz: () => popRateHz,
    raster: () => raster,
    setStdpEnabled: () => setStdpEnabled,
    simT: () => simT,
    stdpEnabled: () => stdpEnabled,
    step: () => step,
    totalSpikes: () => totalSpikes,
    weightHistory: () => weightHistory
  });

  // packages/types/src/neuron.ts
  var MODEL_IDS = {
    LIF: 0,
    ADEX: 1,
    IZHIKEVICH: 2
    // V3+
  };
  var ADEX_STATE = {
    V: 0,
    // Membrane voltage (mV)
    W: 1,
    // Adaptation current (pA)
    GE: 2,
    // Excitatory conductance (nS)
    GI: 3
    // Inhibitory conductance (nS)
  };
  var ADEX_STATE_SIZE = 4;
  var ADEX_PARAMS = {
    CM: 0,
    // Membrane capacitance (pF)
    GL: 1,
    // Leak conductance (nS)
    EL: 2,
    // Leak reversal potential (mV)
    VT: 3,
    // Spike threshold (mV)
    DELTA_T: 4,
    // Slope factor (mV)
    V_PEAK: 5,
    // Spike detection threshold (mV)
    TAU_W: 6,
    // Adaptation time constant (ms)
    A: 7,
    // Subthreshold adaptation coupling (nS)
    B: 8,
    // Spike-triggered adaptation increment (pA)
    V_RESET: 9,
    // Reset potential (mV)
    T_REF: 10,
    // Absolute refractory period (ms)
    EE: 11,
    // Excitatory reversal potential (mV)
    EI: 12
    // Inhibitory reversal potential (mV)
  };
  var ADEX_PARAMS_SIZE = 13;
  var LIF_STATE = {
    V: 0,
    GE: 1,
    GI: 2
  };
  var LIF_STATE_SIZE = 3;
  var LIF_PARAMS = {
    CM: 0,
    GL: 1,
    EL: 2,
    VT: 3,
    V_RESET: 4,
    T_REF: 5,
    EE: 6,
    EI: 7
  };
  var LIF_PARAMS_SIZE = 8;

  // packages/types/src/plasticity.ts
  var DEFAULT_NEUROMODULATOR_STATE = {
    dopamine: 0,
    acetylcholine: 0.5,
    serotonin: 0.5,
    norepinephrine: 0.5
  };
  var NO_UPDATE = { delta: 0 };
  var DEFAULT_STDP_CONFIG = {
    aPlus: 0.01,
    aMinus: 0.0105,
    tauPlus: 20,
    tauMinus: 20,
    wMin: 0,
    wMax: 1
  };

  // packages/config/src/defaults.ts
  function defaultAdExParams() {
    const p = new Float64Array(ADEX_PARAMS_SIZE);
    p[ADEX_PARAMS.CM] = 200;
    p[ADEX_PARAMS.GL] = 10;
    p[ADEX_PARAMS.EL] = -70;
    p[ADEX_PARAMS.VT] = -50;
    p[ADEX_PARAMS.DELTA_T] = 2;
    p[ADEX_PARAMS.V_PEAK] = 0;
    p[ADEX_PARAMS.TAU_W] = 200;
    p[ADEX_PARAMS.A] = 2;
    p[ADEX_PARAMS.B] = 0;
    p[ADEX_PARAMS.V_RESET] = -58;
    p[ADEX_PARAMS.T_REF] = 2;
    p[ADEX_PARAMS.EE] = 0;
    p[ADEX_PARAMS.EI] = -70;
    return p;
  }
  function fastSpikingAdExParams() {
    const p = defaultAdExParams();
    p[ADEX_PARAMS.CM] = 100;
    p[ADEX_PARAMS.GL] = 20;
    p[ADEX_PARAMS.VT] = -47;
    p[ADEX_PARAMS.DELTA_T] = 0.5;
    p[ADEX_PARAMS.TAU_W] = 100;
    p[ADEX_PARAMS.A] = 0;
    p[ADEX_PARAMS.B] = 0;
    p[ADEX_PARAMS.V_RESET] = -65;
    p[ADEX_PARAMS.T_REF] = 1;
    return p;
  }
  function defaultLIFParams() {
    const p = new Float64Array(LIF_PARAMS_SIZE);
    p[LIF_PARAMS.CM] = 200;
    p[LIF_PARAMS.GL] = 10;
    p[LIF_PARAMS.EL] = -70;
    p[LIF_PARAMS.VT] = -50;
    p[LIF_PARAMS.V_RESET] = -65;
    p[LIF_PARAMS.T_REF] = 2;
    p[LIF_PARAMS.EE] = 0;
    p[LIF_PARAMS.EI] = -70;
    return p;
  }
  var TAU_AMPA = 5;
  var TAU_GABA_A = 10;

  // packages/config/src/distributions.ts
  var SeededRNG = class {
    state;
    constructor(seed = 42) {
      this.state = seed >>> 0;
    }
    /** Uniform random float in [0, 1) */
    random() {
      this.state = Math.imul(1664525, this.state) + 1013904223 >>> 0;
      return this.state / 4294967296;
    }
    /** Uniform random float in [min, max) */
    uniform(min, max) {
      return min + this.random() * (max - min);
    }
    /** Standard normal via Box-Muller transform */
    normal(mean = 0, std = 1) {
      const u1 = this.random();
      const u2 = this.random();
      const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
      return mean + std * z;
    }
    /** Lognormal distribution — appropriate for synaptic weight distributions */
    lognormal(mu, sigma) {
      return Math.exp(this.normal(mu, sigma));
    }
    /** Positive normal, clamped to min — for parameters that must be positive */
    positiveNormal(mean, cv = 0.1, min = mean * 0.1) {
      return Math.max(min, this.normal(mean, mean * cv));
    }
    /** Random integer in [min, max] inclusive */
    randint(min, max) {
      return Math.floor(this.uniform(min, max + 1));
    }
    /** Fisher-Yates shuffle of an array (in-place) */
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = this.randint(0, i);
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
      return arr;
    }
  };
  function sampleWeights(dist, n, rng2) {
    const weights = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      switch (dist.type) {
        case "constant":
          weights[i] = dist.value;
          break;
        case "uniform":
          weights[i] = rng2.uniform(dist.min, dist.max);
          break;
        case "normal":
          weights[i] = Math.max(0, rng2.normal(dist.mean, dist.std));
          break;
        case "lognormal":
          weights[i] = rng2.lognormal(dist.mu, dist.sigma);
          break;
      }
    }
    return weights;
  }
  function sampleDelays(dist, n, rng2, minDelay = 0.1) {
    const delays = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      switch (dist.type) {
        case "constant":
          delays[i] = Math.max(minDelay, dist.value);
          break;
        case "uniform":
          delays[i] = Math.max(minDelay, rng2.uniform(dist.min, dist.max));
          break;
      }
    }
    return delays;
  }

  // packages/neurons/src/adex.ts
  var AdExModel = class {
    modelId = MODEL_IDS.ADEX;
    name = "AdEx";
    stateVectorSize = ADEX_STATE_SIZE;
    parameterVectorSize = ADEX_PARAMS_SIZE;
    parameterSchema = {
      C_m: { min: 10, max: 1e3, default: 200, unit: "pF", description: "Membrane capacitance" },
      g_L: { min: 1, max: 100, default: 10, unit: "nS", description: "Leak conductance" },
      E_L: { min: -90, max: -50, default: -70, unit: "mV", description: "Resting/leak reversal potential" },
      V_T: { min: -70, max: -35, default: -50, unit: "mV", description: "Spike threshold" },
      delta_T: { min: 0.1, max: 10, default: 2, unit: "mV", description: "Sharpness of spike initiation" },
      V_peak: { min: -10, max: 40, default: 0, unit: "mV", description: "Spike detection threshold" },
      tau_w: { min: 1, max: 1e3, default: 200, unit: "ms", description: "Adaptation time constant" },
      a: { min: -10, max: 50, default: 2, unit: "nS", description: "Subthreshold adaptation coupling" },
      b: { min: 0, max: 500, default: 0, unit: "pA", description: "Spike-triggered adaptation increment" },
      V_reset: { min: -90, max: -40, default: -58, unit: "mV", description: "Post-spike reset potential" },
      t_ref: { min: 0, max: 20, default: 2, unit: "ms", description: "Absolute refractory period" },
      E_E: { min: -20, max: 20, default: 0, unit: "mV", description: "Excitatory reversal potential" },
      E_I: { min: -100, max: -40, default: -70, unit: "mV", description: "Inhibitory reversal potential" }
    };
    step(state, params, externalCurrent, dt, _t, refractoryRemaining) {
      const V = state[ADEX_STATE.V];
      const w = state[ADEX_STATE.W];
      const gE = state[ADEX_STATE.GE];
      const gI = state[ADEX_STATE.GI];
      const Cm = params[ADEX_PARAMS.CM];
      const gL = params[ADEX_PARAMS.GL];
      const EL = params[ADEX_PARAMS.EL];
      const VT = params[ADEX_PARAMS.VT];
      const deltaT = params[ADEX_PARAMS.DELTA_T];
      const Vpeak = params[ADEX_PARAMS.V_PEAK];
      const tauW = params[ADEX_PARAMS.TAU_W];
      const a = params[ADEX_PARAMS.A];
      const EE = params[ADEX_PARAMS.EE];
      const EI = params[ADEX_PARAMS.EI];
      const gE_new = gE * Math.exp(-dt / TAU_AMPA);
      const gI_new = gI * Math.exp(-dt / TAU_GABA_A);
      if (refractoryRemaining > 0) {
        state[ADEX_STATE.GE] = gE_new;
        state[ADEX_STATE.GI] = gI_new;
        return { spiked: false, voltage: V, adaptation: w };
      }
      const I_syn = gE * (EE - V) + gI * (EI - V);
      const I_total = I_syn + externalCurrent - w;
      const expTerm = gL * deltaT * Math.exp(Math.min((V - VT) / deltaT, 20));
      const dVdt = (-gL * (V - EL) + expTerm + I_total) / Cm;
      const V_new = V + dVdt * dt;
      const dWdt = (a * (V - EL) - w) / tauW;
      const w_new = w + dWdt * dt;
      state[ADEX_STATE.GE] = gE_new;
      state[ADEX_STATE.GI] = gI_new;
      if (V_new >= Vpeak) {
        state[ADEX_STATE.V] = Vpeak;
        state[ADEX_STATE.W] = w_new;
        return { spiked: true, voltage: Vpeak, adaptation: w_new };
      }
      state[ADEX_STATE.V] = V_new;
      state[ADEX_STATE.W] = w_new;
      return { spiked: false, voltage: V_new, adaptation: w_new };
    }
    resetState(state, params) {
      const Vreset = params[ADEX_PARAMS.V_RESET];
      const b = params[ADEX_PARAMS.B];
      const w = state[ADEX_STATE.W];
      state[ADEX_STATE.V] = Vreset;
      state[ADEX_STATE.W] = w + b;
    }
    initState(params) {
      const state = new Float64Array(ADEX_STATE_SIZE);
      state[ADEX_STATE.V] = params[ADEX_PARAMS.EL];
      state[ADEX_STATE.W] = 0;
      state[ADEX_STATE.GE] = 0;
      state[ADEX_STATE.GI] = 0;
      return state;
    }
    defaultParams() {
      return defaultAdExParams();
    }
    serializeState(state) {
      return new Uint8Array(state.buffer.slice(state.byteOffset, state.byteOffset + state.byteLength));
    }
    deserializeState(data) {
      return new Float64Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    }
  };

  // packages/neurons/src/lif.ts
  var LIFModel = class {
    modelId = MODEL_IDS.LIF;
    name = "LIF";
    stateVectorSize = LIF_STATE_SIZE;
    parameterVectorSize = LIF_PARAMS_SIZE;
    parameterSchema = {
      C_m: { min: 10, max: 1e3, default: 200, unit: "pF", description: "Membrane capacitance" },
      g_L: { min: 1, max: 100, default: 10, unit: "nS", description: "Leak conductance" },
      E_L: { min: -90, max: -50, default: -70, unit: "mV", description: "Resting potential" },
      V_T: { min: -70, max: -35, default: -50, unit: "mV", description: "Spike threshold" },
      V_reset: { min: -90, max: -40, default: -65, unit: "mV", description: "Reset potential" },
      t_ref: { min: 0, max: 20, default: 2, unit: "ms", description: "Refractory period" },
      E_E: { min: -20, max: 20, default: 0, unit: "mV", description: "Excitatory reversal" },
      E_I: { min: -100, max: -40, default: -70, unit: "mV", description: "Inhibitory reversal" }
    };
    step(state, params, externalCurrent, dt, _t, refractoryRemaining) {
      const V = state[LIF_STATE.V];
      const gE = state[LIF_STATE.GE];
      const gI = state[LIF_STATE.GI];
      const Cm = params[LIF_PARAMS.CM];
      const gL = params[LIF_PARAMS.GL];
      const EL = params[LIF_PARAMS.EL];
      const VT = params[LIF_PARAMS.VT];
      const EE = params[LIF_PARAMS.EE];
      const EI = params[LIF_PARAMS.EI];
      const gE_new = gE * Math.exp(-dt / TAU_AMPA);
      const gI_new = gI * Math.exp(-dt / TAU_GABA_A);
      state[LIF_STATE.GE] = gE_new;
      state[LIF_STATE.GI] = gI_new;
      if (refractoryRemaining > 0) {
        return { spiked: false, voltage: V, adaptation: 0 };
      }
      const I_syn = gE * (EE - V) + gI * (EI - V);
      const dVdt = (-gL * (V - EL) + I_syn + externalCurrent) / Cm;
      const V_new = V + dVdt * dt;
      if (V_new >= VT) {
        state[LIF_STATE.V] = VT;
        return { spiked: true, voltage: VT, adaptation: 0 };
      }
      state[LIF_STATE.V] = V_new;
      return { spiked: false, voltage: V_new, adaptation: 0 };
    }
    resetState(state, params) {
      state[LIF_STATE.V] = params[LIF_PARAMS.V_RESET];
    }
    initState(params) {
      const state = new Float64Array(LIF_STATE_SIZE);
      state[LIF_STATE.V] = params[LIF_PARAMS.EL];
      state[LIF_STATE.GE] = 0;
      state[LIF_STATE.GI] = 0;
      return state;
    }
    defaultParams() {
      return defaultLIFParams();
    }
    serializeState(state) {
      return new Uint8Array(state.buffer.slice(state.byteOffset, state.byteOffset + state.byteLength));
    }
    deserializeState(data) {
      return new Float64Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
    }
  };

  // packages/neurons/src/registry.ts
  var _registry = /* @__PURE__ */ new Map();
  function registerModel(model) {
    if (_registry.has(model.modelId)) {
      throw new Error(`Model ID ${model.modelId} (${model.name}) is already registered`);
    }
    _registry.set(model.modelId, model);
  }
  registerModel(new AdExModel());
  registerModel(new LIFModel());

  // packages/neurons/src/population-stepper.ts
  function readNeuronState(state, neuronIndex, n, stateSize) {
    const slice = new Float64Array(stateSize);
    for (let s = 0; s < stateSize; s++) {
      slice[s] = state[s * n + neuronIndex];
    }
    return slice;
  }
  function writeNeuronState(state, neuronIndex, n, slice) {
    for (let s = 0; s < slice.length; s++) {
      state[s * n + neuronIndex] = slice[s];
    }
  }
  function readNeuronParams(params, neuronIndex, n, paramSize) {
    const slice = new Float64Array(paramSize);
    for (let p = 0; p < paramSize; p++) {
      slice[p] = params[p * n + neuronIndex];
    }
    return slice;
  }
  function stepPopulation(population, model, currents, dt, t) {
    const n = population.config.size;
    const stateSize = model.stateVectorSize;
    const paramSize = model.parameterVectorSize;
    const spikeIndices = [];
    for (let i = 0; i < n; i++) {
      const neuronState = readNeuronState(population.state, i, n, stateSize);
      const neuronParams = readNeuronParams(population.params, i, n, paramSize);
      const refractoryRemaining = population.refractory[i];
      const current = currents[i];
      const result = model.step(neuronState, neuronParams, current, dt, t, refractoryRemaining);
      if (result.spiked) {
        model.resetState(neuronState, neuronParams);
        const tRef = neuronParams[10];
        population.refractory[i] = tRef;
        spikeIndices.push(i);
        population.spikeCounts[i] = population.spikeCounts[i] + 1;
      } else if (refractoryRemaining > 0) {
        population.refractory[i] = Math.max(0, refractoryRemaining - dt);
      }
      writeNeuronState(population.state, i, n, neuronState);
    }
    return { spikeIndices };
  }
  function initializePopulationState(population, model) {
    const n = population.config.size;
    const paramSize = model.parameterVectorSize;
    for (let i = 0; i < n; i++) {
      const neuronParams = readNeuronParams(population.params, i, n, paramSize);
      const initialState = model.initState(neuronParams);
      writeNeuronState(population.state, i, n, initialState);
    }
  }

  // packages/network/src/csr-store.ts
  var RECEPTOR_IDS = {
    AMPA: 0,
    GABA_A: 1,
    NMDA: 2,
    GABA_B: 3
  };
  var RECEPTOR_FROM_ID = ["AMPA", "GABA_A", "NMDA", "GABA_B"];
  var CSRConnectivityStore = class {
    sourceSize;
    targetSize;
    rowPtr;
    colIdx;
    _weights;
    _delays;
    _receptor;
    get synapseCount() {
      return this.colIdx.length;
    }
    constructor(sourceSize, targetSize, rowPtr, colIdx, weights, delays, receptor) {
      this.sourceSize = sourceSize;
      this.targetSize = targetSize;
      this.rowPtr = rowPtr;
      this.colIdx = colIdx;
      this._weights = weights;
      this._delays = delays;
      this._receptor = receptor;
    }
    getOutgoing(sourceIndex) {
      const start = this.rowPtr[sourceIndex];
      const end = this.rowPtr[sourceIndex + 1];
      const results = [];
      for (let idx = start; idx < end; idx++) {
        results.push(this.makeSynapseView(idx, sourceIndex));
      }
      return results;
    }
    getIncoming(targetIndex) {
      const results = [];
      for (let src = 0; src < this.sourceSize; src++) {
        const start = this.rowPtr[src];
        const end = this.rowPtr[src + 1];
        for (let idx = start; idx < end; idx++) {
          if (this.colIdx[idx] === targetIndex) {
            results.push(this.makeSynapseView(idx, src));
          }
        }
      }
      return results;
    }
    forEach(callback) {
      for (let src = 0; src < this.sourceSize; src++) {
        const start = this.rowPtr[src];
        const end = this.rowPtr[src + 1];
        for (let idx = start; idx < end; idx++) {
          callback(this.makeSynapseView(idx, src));
        }
      }
    }
    getWeight(synapseIndex) {
      return this._weights[synapseIndex];
    }
    setWeight(synapseIndex, weight) {
      this._weights[synapseIndex] = weight;
    }
    clampWeights(minWeight, maxWeight) {
      for (let i = 0; i < this._weights.length; i++) {
        this._weights[i] = Math.max(minWeight, Math.min(maxWeight, this._weights[i]));
      }
    }
    getWeightsSnapshot() {
      return this._weights.slice();
    }
    getDelay(synapseIndex) {
      return this._delays[synapseIndex];
    }
    getReceptorType(synapseIndex) {
      return RECEPTOR_FROM_ID[this._receptor[synapseIndex]] ?? "AMPA";
    }
    makeSynapseView(idx, sourceIndex) {
      const store = this;
      return {
        index: idx,
        sourceIndex,
        targetIndex: store.colIdx[idx],
        delay: store._delays[idx],
        receptorType: RECEPTOR_FROM_ID[store._receptor[idx]] ?? "AMPA",
        get weight() {
          return store._weights[idx];
        },
        set weight(v) {
          store._weights[idx] = v;
        }
      };
    }
  };
  function buildCSRStore(sourceSize, targetSize, rule, weightDist, delayDist, receptorType, rng2) {
    const srcs = [];
    const tgts = [];
    switch (rule.type) {
      case "all_to_all":
        for (let s = 0; s < sourceSize; s++) {
          for (let t = 0; t < targetSize; t++) {
            srcs.push(s);
            tgts.push(t);
          }
        }
        break;
      case "one_to_one": {
        const n = Math.min(sourceSize, targetSize);
        for (let i = 0; i < n; i++) {
          srcs.push(i);
          tgts.push(i);
        }
        break;
      }
      case "random": {
        const localRng = rule.seed !== void 0 ? new SeededRNG(rule.seed) : rng2;
        for (let s = 0; s < sourceSize; s++) {
          for (let t = 0; t < targetSize; t++) {
            if (localRng.random() < rule.probability) {
              srcs.push(s);
              tgts.push(t);
            }
          }
        }
        break;
      }
      case "fixed_number_pre": {
        const localRng = rule.seed !== void 0 ? new SeededRNG(rule.seed) : rng2;
        const allTargets = Array.from({ length: targetSize }, (_, i) => i);
        for (let s = 0; s < sourceSize; s++) {
          const n = Math.min(rule.n, targetSize);
          const chosen = localRng.shuffle([...allTargets]).slice(0, n);
          for (const t of chosen) {
            srcs.push(s);
            tgts.push(t);
          }
        }
        break;
      }
      case "custom":
        for (const [s, t] of rule.connections) {
          srcs.push(s);
          tgts.push(t);
        }
        break;
      case "topographic":
        for (let s = 0; s < sourceSize; s++) {
          for (let t = 0; t < targetSize; t++) {
            const dist = Math.abs(s / sourceSize - t / targetSize);
            const prob = Math.exp(-(dist * dist) / (2 * rule.sigma * rule.sigma));
            if (rng2.random() < prob) {
              srcs.push(s);
              tgts.push(t);
            }
          }
        }
        break;
    }
    const S = srcs.length;
    const weights = sampleWeights(weightDist, S, rng2);
    const delays = sampleDelays(delayDist, S, rng2);
    const receptor = new Uint8Array(S).fill(RECEPTOR_IDS[receptorType]);
    const order = Array.from({ length: S }, (_, i) => i).sort((a, b) => srcs[a] - srcs[b]);
    const sortedSrcs = order.map((i) => srcs[i]);
    const sortedTgts = new Int32Array(order.map((i) => tgts[i]));
    const sortedWeights = new Float32Array(order.map((i) => weights[i]));
    const sortedDelays = new Float32Array(order.map((i) => delays[i]));
    const sortedReceptor = new Uint8Array(order.map((i) => receptor[i]));
    const rowPtr = new Int32Array(sourceSize + 1);
    for (const src of sortedSrcs) {
      rowPtr[src + 1]++;
    }
    for (let i = 1; i <= sourceSize; i++) {
      rowPtr[i] += rowPtr[i - 1];
    }
    return new CSRConnectivityStore(
      sourceSize,
      targetSize,
      rowPtr,
      sortedTgts,
      sortedWeights,
      sortedDelays,
      sortedReceptor
    );
  }

  // packages/network/src/delay-line.ts
  var CircularDelayLine = class {
    maxDelaySteps;
    writeHead = 0;
    buffer;
    constructor(maxDelayMs, dt) {
      this.maxDelaySteps = Math.ceil(maxDelayMs / dt) + 1;
      this.buffer = Array.from({ length: this.maxDelaySteps }, () => []);
    }
    enqueue(neuronIndex, delaySteps) {
      const slot = (this.writeHead + delaySteps) % this.maxDelaySteps;
      this.buffer[slot].push(neuronIndex);
    }
    dequeue() {
      const current = this.buffer[this.writeHead];
      return current;
    }
    advance() {
      this.buffer[this.writeHead].length = 0;
      this.writeHead = (this.writeHead + 1) % this.maxDelaySteps;
    }
    /** How many spikes are queued across all slots (for diagnostic purposes) */
    totalQueued() {
      return this.buffer.reduce((sum, slot) => sum + slot.length, 0);
    }
  };

  // packages/network/src/conductance.ts
  function getConductanceIndex(receptorType, modelId) {
    if (modelId === MODEL_IDS.ADEX) {
      switch (receptorType) {
        case "AMPA":
        case "NMDA":
          return ADEX_STATE.GE;
        case "GABA_A":
        case "GABA_B":
          return ADEX_STATE.GI;
      }
    } else if (modelId === MODEL_IDS.LIF) {
      switch (receptorType) {
        case "AMPA":
        case "NMDA":
          return LIF_STATE.GE;
        case "GABA_A":
        case "GABA_B":
          return LIF_STATE.GI;
      }
    }
    return 0;
  }
  function applySpikeToTarget(population, targetIndex, weight, receptorType, stateSize) {
    const n = population.config.size;
    const modelId = population.config.modelId;
    const gIdx = getConductanceIndex(receptorType, modelId);
    const stateIdx = gIdx * n + targetIndex;
    population.state[stateIdx] = population.state[stateIdx] + weight;
  }
  function deliverSpikes(spikeIndices, target, store, receptorType, stateSize) {
    for (const srcIdx of spikeIndices) {
      const outgoing = store.getOutgoing(srcIdx);
      for (const syn of outgoing) {
        applySpikeToTarget(target, syn.targetIndex, syn.weight, receptorType, stateSize);
      }
    }
  }

  // packages/plasticity/src/stdp.ts
  var STDPRule = class {
    ruleId = "stdp-nearest-neighbor";
    description = "Nearest-neighbor STDP: LTP on pre\u2192post, LTD on post\u2192pre";
    requiredTraces = ["pre", "post"];
    config;
    constructor(config = {}) {
      this.config = { ...DEFAULT_STDP_CONFIG, ...config };
    }
    /**
     * Called when a pre-synaptic neuron fires.
     * Implements LTD: post-synaptic trace o_j captures recent post-synaptic activity.
     * If post fired recently (high o_j), pre→post ordering means post fired BEFORE pre → LTD.
     */
    onPreSpike(_synapseIndex, _preTrace, postTrace, _eligibility, currentWeight, _context) {
      const delta = -this.config.aMinus * postTrace;
      return delta !== 0 ? { delta } : NO_UPDATE;
    }
    /**
     * Called when a post-synaptic neuron fires.
     * Implements LTP: pre-synaptic trace r_i captures recent pre-synaptic activity.
     * If pre fired recently (high r_i), pre→post ordering means pre fired BEFORE post → LTP.
     */
    onPostSpike(_synapseIndex, preTrace, _postTrace, _eligibility, currentWeight, _context) {
      const delta = this.config.aPlus * preTrace;
      return delta !== 0 ? { delta } : NO_UPDATE;
    }
    getConfig() {
      return this.config;
    }
  };

  // packages/observability/src/event-bus.ts
  var SpikeEventBus = class {
    subscribers = /* @__PURE__ */ new Map();
    get subscriptionCount() {
      let count = 0;
      for (const subs of this.subscribers.values()) {
        count += subs.length;
      }
      return count;
    }
    subscribe(eventType, handler, filter) {
      if (!this.subscribers.has(eventType)) {
        this.subscribers.set(eventType, []);
      }
      const sub = { handler, filter };
      this.subscribers.get(eventType).push(sub);
      return () => {
        const subs = this.subscribers.get(eventType);
        if (subs) {
          const idx = subs.indexOf(sub);
          if (idx !== -1) subs.splice(idx, 1);
        }
      };
    }
    emit(event) {
      const subs = this.subscribers.get(event.type);
      if (!subs || subs.length === 0) return;
      for (const sub of subs) {
        if (this.matchesFilter(event, sub.filter)) {
          sub.handler(event);
        }
      }
    }
    clear() {
      this.subscribers.clear();
    }
    matchesFilter(event, filter) {
      if (!filter) return true;
      if (filter.populationId !== void 0) {
        if ("populationId" in event && event.populationId !== filter.populationId) {
          return false;
        }
      }
      if (filter.projectionId !== void 0) {
        if ("projectionId" in event && event.projectionId !== filter.projectionId) {
          return false;
        }
      }
      return true;
    }
  };

  // demo/main.ts
  var N_E = 80;
  var N_I = 20;
  var DT = 0.1;
  var MAX_DELAY_MS = 5;
  var W_MIN = 0;
  var W_MAX = 6;
  var STDP_RULE = new STDPRule({ aPlus: 0.01, aMinus: 0.012, wMin: W_MIN, wMax: W_MAX });
  var DECAY_PRE = Math.exp(-DT / 20);
  var DECAY_POST = Math.exp(-DT / 20);
  function makeCtx(t) {
    return {
      t,
      dt: DT,
      neuromodulators: DEFAULT_NEUROMODULATOR_STATE,
      homeostatic: {
        targetRate: 5,
        currentRates: new Float32Array(1),
        scalingFactors: new Float32Array(1).fill(1),
        excitabilityOffsets: new Float32Array(1)
      }
    };
  }
  var CTX = makeCtx(0);
  var bus = new SpikeEventBus();
  var raster = [];
  var weightHistory = [];
  var simT = 0;
  var totalSpikes = 0;
  var stdpEnabled = true;
  var excPop;
  var inhPop;
  var rng;
  var storeEE;
  var storeEI;
  var storeIE;
  var storeII;
  var dlEE;
  var dlEI;
  var dlIE;
  var dlII;
  var preTraceE;
  var postTraceE;
  var currentsE;
  var currentsI;
  var adex = new AdExModel();
  var weightSampleCounter = 0;
  function makePop(id, size, role) {
    return {
      id,
      config: { name: role, size, modelId: MODEL_IDS.ADEX, role },
      state: new Float64Array(size * ADEX_STATE_SIZE),
      params: new Float64Array(size * ADEX_PARAMS_SIZE),
      refractory: new Float64Array(size),
      spikeCounts: new Float64Array(size)
    };
  }
  function fillParamsSoA(pop, baseParams) {
    const n = pop.config.size;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < ADEX_PARAMS_SIZE; k++) {
        const base = baseParams[k];
        const noise = 1 + rng.normal(0, 0.05);
        pop.params[k * n + i] = base * noise;
      }
    }
  }
  function init(seed = 42) {
    rng = new SeededRNG(seed);
    simT = 0;
    totalSpikes = 0;
    raster.length = 0;
    excPop = makePop(0, N_E, "excitatory");
    inhPop = makePop(1, N_I, "inhibitory");
    fillParamsSoA(excPop, defaultAdExParams());
    fillParamsSoA(inhPop, fastSpikingAdExParams());
    initializePopulationState(excPop, adex);
    initializePopulationState(inhPop, adex);
    currentsE = new Float64Array(N_E);
    currentsI = new Float64Array(N_I);
    storeEE = buildCSRStore(
      N_E,
      N_E,
      { type: "random", probability: 0.1 },
      { type: "lognormal", mu: Math.log(1.2), sigma: 0.3 },
      { type: "uniform", min: 0.5, max: MAX_DELAY_MS },
      "AMPA",
      rng
    );
    storeEI = buildCSRStore(
      N_E,
      N_I,
      { type: "random", probability: 0.12 },
      { type: "lognormal", mu: Math.log(0.8), sigma: 0.3 },
      { type: "uniform", min: 0.5, max: MAX_DELAY_MS },
      "AMPA",
      rng
    );
    storeIE = buildCSRStore(
      N_I,
      N_E,
      { type: "random", probability: 0.35 },
      { type: "lognormal", mu: Math.log(2.5), sigma: 0.3 },
      { type: "uniform", min: 0.5, max: MAX_DELAY_MS },
      "GABA_A",
      rng
    );
    storeII = buildCSRStore(
      N_I,
      N_I,
      { type: "random", probability: 0.15 },
      { type: "lognormal", mu: Math.log(1), sigma: 0.3 },
      { type: "uniform", min: 0.5, max: MAX_DELAY_MS },
      "GABA_A",
      rng
    );
    dlEE = new CircularDelayLine(MAX_DELAY_MS, DT);
    dlEI = new CircularDelayLine(MAX_DELAY_MS, DT);
    dlIE = new CircularDelayLine(MAX_DELAY_MS, DT);
    dlII = new CircularDelayLine(MAX_DELAY_MS, DT);
    preTraceE = new Float32Array(N_E);
    postTraceE = new Float32Array(N_E);
    weightHistory.length = 0;
    weightSampleCounter = 0;
  }
  function step(driveMode) {
    simT += DT;
    CTX = makeCtx(simT);
    deliverSpikes(dlEE.dequeue(), excPop, storeEE, "AMPA", ADEX_STATE_SIZE);
    deliverSpikes(dlEI.dequeue(), inhPop, storeEI, "AMPA", ADEX_STATE_SIZE);
    deliverSpikes(dlIE.dequeue(), excPop, storeIE, "GABA_A", ADEX_STATE_SIZE);
    deliverSpikes(dlII.dequeue(), inhPop, storeII, "GABA_A", ADEX_STATE_SIZE);
    const Iext = driveMode === "tonic" ? 250 : driveMode === "burst" ? simT % 200 < 25 ? 700 : 0 : 0;
    currentIext = Iext;
    currentsE.fill(Iext);
    currentsI.fill(0);
    const { spikeIndices: spkE } = stepPopulation(excPop, adex, currentsE, DT, simT);
    const { spikeIndices: spkI } = stepPopulation(inhPop, adex, currentsI, DT, simT);
    for (const idx of spkE) {
      totalSpikes++;
      raster.push(simT, idx);
      bus.emit({ type: "spike", populationId: 0, neuronIndex: idx, t: simT });
      for (const syn of storeEE.getOutgoing(idx))
        dlEE.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
      for (const syn of storeEI.getOutgoing(idx))
        dlEI.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
    }
    for (const idx of spkI) {
      totalSpikes++;
      raster.push(simT, N_E + idx);
      bus.emit({ type: "spike", populationId: 1, neuronIndex: idx, t: simT });
      for (const syn of storeIE.getOutgoing(idx))
        dlIE.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
      for (const syn of storeII.getOutgoing(idx))
        dlII.enqueue(idx, Math.max(1, Math.round(syn.delay / DT)));
    }
    if (stdpEnabled) {
      for (let i = 0; i < N_E; i++) {
        preTraceE[i] = preTraceE[i] * DECAY_PRE;
        postTraceE[i] = postTraceE[i] * DECAY_POST;
      }
      for (const pre of spkE) {
        for (const syn of storeEE.getOutgoing(pre)) {
          const upd = STDP_RULE.onPreSpike(
            syn.index,
            preTraceE[pre],
            postTraceE[syn.targetIndex],
            0,
            syn.weight,
            CTX
          );
          if (upd.delta !== 0)
            storeEE.setWeight(syn.index, Math.max(W_MIN, Math.min(W_MAX, syn.weight + upd.delta)));
        }
        preTraceE[pre] = preTraceE[pre] + 1;
      }
      for (const post of spkE) {
        for (const syn of storeEE.getIncoming(post)) {
          const upd = STDP_RULE.onPostSpike(
            syn.index,
            preTraceE[syn.sourceIndex],
            postTraceE[post],
            0,
            syn.weight,
            CTX
          );
          if (upd.delta !== 0)
            storeEE.setWeight(syn.index, Math.max(W_MIN, Math.min(W_MAX, syn.weight + upd.delta)));
        }
        postTraceE[post] = postTraceE[post] + 1;
      }
    }
    if (++weightSampleCounter % 1e3 === 0) {
      weightHistory.push(meanEEWeight());
      if (weightHistory.length > 600) weightHistory.shift();
    }
    dlEE.advance();
    dlEI.advance();
    dlIE.advance();
    dlII.advance();
    if (raster.length > 2e4) {
      const tCut = simT - 600;
      let start = 0;
      while (start < raster.length && raster[start] < tCut) start += 2;
      if (start > 0) raster.splice(0, start);
    }
  }
  function meanEEWeight() {
    const snap = storeEE.getWeightsSnapshot();
    if (!snap.length) return 0;
    let s = 0;
    for (let i = 0; i < snap.length; i++) s += snap[i];
    return s / snap.length;
  }
  function popRateHz(windowMs = 200) {
    const tMin = simT - windowMs;
    let n = 0;
    for (let k = 0; k < raster.length; k += 2)
      if (raster[k] >= tMin) n++;
    return n / (N_E + N_I) / (windowMs / 1e3);
  }
  var N_TOTAL = N_E + N_I;
  var N_EXC = N_E;
  var currentIext = 0;
  function setStdpEnabled(v) {
    stdpEnabled = v;
  }
  return __toCommonJS(main_exports);
})();
