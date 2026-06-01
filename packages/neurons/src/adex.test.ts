/**
 * AdEx model tests — verified against analytical and numerical solutions.
 *
 * Tests must pass for V1 to be considered correct. These serve as the
 * reference for validating the Rust/WASM implementation in V2.
 */

import { AdExModel } from './adex';
import { ADEX_STATE, ADEX_PARAMS, ADEX_STATE_SIZE } from '@snn/types';
import { defaultAdExParams, fastSpikingAdExParams, burstingAdExParams } from '@snn/config';

describe('AdExModel', () => {
  let model: AdExModel;
  let params: Float64Array;
  let state: Float64Array;

  beforeEach(() => {
    model = new AdExModel();
    params = defaultAdExParams();
    state = model.initState(params);
  });

  test('resting state has voltage at E_L', () => {
    expect(state[ADEX_STATE.V]).toBeCloseTo(params[ADEX_PARAMS.EL] as number, 5);
    expect(state[ADEX_STATE.W]).toBeCloseTo(0, 5);
    expect(state[ADEX_STATE.GE]).toBeCloseTo(0, 5);
    expect(state[ADEX_STATE.GI]).toBeCloseTo(0, 5);
  });

  test('subthreshold current does not spike', () => {
    // Inject 50pA — below threshold for default params
    const result = model.step(state, params, 50, 0.1, 0, 0);
    expect(result.spiked).toBe(false);
    expect(result.voltage).toBeGreaterThan(params[ADEX_PARAMS.EL] as number);
  });

  test('suprathreshold current triggers spike', () => {
    // Set voltage just below V_PEAK (0mV) — spike fires when V >= V_PEAK
    // With V = -1mV and large current, the exponential term will push it over in one step
    state[ADEX_STATE.V] = -2.0; // Just below V_PEAK = 0mV
    const result = model.step(state, params, 2000, 0.1, 0, 0);
    expect(result.spiked).toBe(true);
  });

  test('reset applies correctly after spike', () => {
    // Set V above V_PEAK to trigger spike on next step
    state[ADEX_STATE.V] = -2.0;
    state[ADEX_STATE.W] = 0; // Start with zero adaptation for predictable reset
    model.step(state, params, 2000, 0.1, 0, 0);
    // W evolves slightly during the step — capture it before resetState
    const wAfterStep = state[ADEX_STATE.W] as number;
    model.resetState(state, params);
    expect(state[ADEX_STATE.V]).toBeCloseTo(params[ADEX_PARAMS.V_RESET] as number, 3);
    // resetState adds b to the post-step w value
    expect(state[ADEX_STATE.W]).toBeCloseTo(wAfterStep + (params[ADEX_PARAMS.B] as number), 3);
  });

  test('adaptation increment b increases w after spike for bursting neurons', () => {
    const burstParams = burstingAdExParams();
    const burstState = model.initState(burstParams);
    burstState[ADEX_STATE.V] = -2.0; // Above V_PEAK threshold
    burstState[ADEX_STATE.W] = 0;
    model.step(burstState, burstParams, 2000, 0.1, 0, 0);
    const wBeforeReset = burstState[ADEX_STATE.W] as number;
    model.resetState(burstState, burstParams);
    const b = burstParams[ADEX_PARAMS.B] as number;
    // After reset: w = w_after_step + b; b=80 for bursting neuron
    expect(b).toBeGreaterThan(0);
    expect(burstState[ADEX_STATE.W]).toBeCloseTo(wBeforeReset + b, 3);
  });

  test('refractory period clamps dynamics', () => {
    const V_before = (params[ADEX_PARAMS.V_RESET] as number);
    state[ADEX_STATE.V] = V_before;
    // With refractoryRemaining > 0, voltage should not change from membrane dynamics
    const result = model.step(state, params, 1000, 0.1, 0, 2.0);
    expect(result.spiked).toBe(false);
    expect(result.voltage).toBeCloseTo(V_before, 5);
  });

  test('conductances decay exponentially', () => {
    state[ADEX_STATE.GE] = 10.0; // nS
    state[ADEX_STATE.GI] = 5.0;
    const dt = 0.1;
    model.step(state, params, 0, dt, 0, 0);
    // AMPA tau = 5ms: expected gE after dt=0.1ms
    const expectedGE = 10.0 * Math.exp(-dt / 5.0);
    expect(state[ADEX_STATE.GE]).toBeCloseTo(expectedGE, 5);
  });

  test('GABA-A conductance produces hyperpolarizing current', () => {
    // Set inhibitory conductance
    state[ADEX_STATE.V] = -60.0; // Above E_I = -70mV
    state[ADEX_STATE.GI] = 20.0;
    // Expected synaptic current: I = gI * (EI - V) = 20 * (-70 - (-60)) = -200 pA (hyperpolarizing)
    const result = model.step(state, params, 0, 0.1, 0, 0);
    // Voltage should move toward E_I = -70mV
    expect(result.voltage).toBeLessThan(-60.0);
  });

  test('fast spiking params produce shorter membrane time constant', () => {
    // FS neurons have higher gl (20 vs 10 nS) so need proportionally more current to spike.
    // Test the key property: τ_m = Cm/gl is shorter for FS (100/20=5ms vs 200/10=20ms).
    const fsParams = fastSpikingAdExParams();
    const regParams = defaultAdExParams();

    const fsTau = (fsParams[ADEX_PARAMS.CM] as number) / (fsParams[ADEX_PARAMS.GL] as number);
    const regTau = (regParams[ADEX_PARAMS.CM] as number) / (regParams[ADEX_PARAMS.GL] as number);
    expect(fsTau).toBeLessThan(regTau);
    expect(fsTau).toBeCloseTo(5.0, 1);   // 100/20 = 5ms
    expect(regTau).toBeCloseTo(20.0, 1); // 200/10 = 20ms

    // Both should spike given sufficient current (proportional to their gl)
    const fsState = model.initState(fsParams);
    const regState = model.initState(regParams);
    let fsSpike = false;
    let regSpike = false;

    for (let step = 0; step < 2000 && (!fsSpike || !regSpike); step++) {
      if (!fsSpike) {
        // FS needs more current (2× higher gl) — use 600pA for FS, 300pA for regular
        const r = model.step(fsState, fsParams, 600, 0.1, step * 0.1, fsSpike ? 2.0 : 0);
        if (r.spiked) { model.resetState(fsState, fsParams); fsSpike = true; }
      }
      if (!regSpike) {
        const r = model.step(regState, regParams, 300, 0.1, step * 0.1, regSpike ? 2.0 : 0);
        if (r.spiked) { model.resetState(regState, regParams); regSpike = true; }
      }
    }
    expect(fsSpike).toBe(true);
    expect(regSpike).toBe(true);
  });

  test('serialization roundtrip preserves state exactly', () => {
    state[ADEX_STATE.V] = -55.0;
    state[ADEX_STATE.W] = 42.3;
    const serialized = model.serializeState(state);
    const restored = model.deserializeState(serialized);
    for (let i = 0; i < ADEX_STATE_SIZE; i++) {
      expect(restored[i]).toBeCloseTo(state[i] as number, 10);
    }
  });

  test('defaultParams passes schema bounds', () => {
    const params = model.defaultParams();
    const schema = model.parameterSchema;
    const names = Object.keys(schema);
    names.forEach((name, i) => {
      const bounds = schema[name]!;
      expect(params[i]).toBeGreaterThanOrEqual(bounds.min);
      expect(params[i]).toBeLessThanOrEqual(bounds.max);
    });
  });
});
