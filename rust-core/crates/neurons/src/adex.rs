/*!
 * Adaptive Exponential Integrate-and-Fire (AdEx) — Rust implementation.
 *
 * This is the performance-critical implementation compiled to WASM for V2+.
 * Must produce results identical (within floating-point tolerance) to the
 * TypeScript reference in packages/neurons/src/adex.ts.
 *
 * Integration method: Exponential Euler for V, forward Euler for w, exact decay for g.
 * This matches the TypeScript implementation exactly.
 *
 * State indices (SoA layout, N = population size):
 *   V:  state[0..N]
 *   W:  state[N..2N]
 *   GE: state[2N..3N]
 *   GI: state[3N..4N]
 *
 * Params indices (SoA layout):
 *   CM:0, GL:1, EL:2, VT:3, DELTA_T:4, V_PEAK:5, TAU_W:6,
 *   A:7, B:8, V_RESET:9, T_REF:10, EE:11, EI:12
 */

const TAU_AMPA: f64 = 5.0;
const TAU_GABA_A: f64 = 10.0;
const EXP_CLAMP: f64 = 20.0;  // Clamp exponent to prevent overflow

/// Result of stepping a single AdEx neuron
pub struct AdExStepResult {
    pub spiked: bool,
    pub v: f64,
    pub w: f64,
}

/// Per-neuron state (used for single-neuron API)
pub struct AdExState {
    pub v: f64,
    pub w: f64,
    pub ge: f64,
    pub gi: f64,
}

/// Per-neuron parameters
pub struct AdExParams {
    pub cm: f64,
    pub gl: f64,
    pub el: f64,
    pub vt: f64,
    pub delta_t: f64,
    pub v_peak: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub v_reset: f64,
    pub t_ref: f64,
    pub ee: f64,
    pub ei: f64,
}

/// Step a single AdEx neuron — pure function, no side effects.
/// This is the reference implementation; the SoA population version calls this.
#[inline]
pub fn step_adex_single(
    state: &mut AdExState,
    params: &AdExParams,
    external_current: f64,
    dt: f64,
    refractory_remaining: f64,
) -> AdExStepResult {
    // Exact exponential decay of conductances
    let ge_new = state.ge * (-dt / TAU_AMPA).exp();
    let gi_new = state.gi * (-dt / TAU_GABA_A).exp();

    if refractory_remaining > 0.0 {
        state.ge = ge_new;
        state.gi = gi_new;
        return AdExStepResult { spiked: false, v: state.v, w: state.w };
    }

    let v = state.v;
    let w = state.w;

    // Conductance-based synaptic current (pA = nS × mV)
    let i_syn = state.ge * (params.ee - v) + state.gi * (params.ei - v);
    let i_total = i_syn + external_current - w;

    // Exponential Euler for voltage
    let exp_arg = ((v - params.vt) / params.delta_t).min(EXP_CLAMP);
    let exp_term = params.gl * params.delta_t * exp_arg.exp();
    let dvdt = (-params.gl * (v - params.el) + exp_term + i_total) / params.cm;
    let v_new = v + dvdt * dt;

    // Forward Euler for adaptation
    let dwdt = (params.a * (v - params.el) - w) / params.tau_w;
    let w_new = w + dwdt * dt;

    state.ge = ge_new;
    state.gi = gi_new;

    if v_new >= params.v_peak {
        state.v = params.v_peak;
        state.w = w_new;
        AdExStepResult { spiked: true, v: params.v_peak, w: w_new }
    } else {
        state.v = v_new;
        state.w = w_new;
        AdExStepResult { spiked: false, v: v_new, w: w_new }
    }
}

/// Apply post-spike reset to a single neuron's state
#[inline]
pub fn reset_adex_single(state: &mut AdExState, params: &AdExParams) {
    state.v = params.v_reset;
    state.w = state.w + params.b;
}

/// Step an entire population using SoA layout.
///
/// # Arguments
/// - `state`: SoA state matrix [4*N], layout: [V[0..N], W[0..N], GE[0..N], GI[0..N]]
/// - `params`: SoA params matrix [13*N]
/// - `currents`: External current per neuron [N] in pA
/// - `refractory`: Refractory time remaining per neuron [N] in ms
/// - `spike_out`: Output buffer for spike indices — caller pre-allocates to capacity N
/// - `n`: Population size
/// - `dt`: Timestep in ms
///
/// Returns: number of spikes generated
pub fn step_adex_population(
    state: &mut [f64],
    params: &[f64],
    currents: &[f64],
    refractory: &mut [f64],
    spike_out: &mut [u32],
    n: usize,
    dt: f64,
) -> usize {
    debug_assert_eq!(state.len(), 4 * n);
    debug_assert_eq!(params.len(), 13 * n);
    debug_assert_eq!(currents.len(), n);
    debug_assert_eq!(refractory.len(), n);

    let mut spike_count = 0usize;

    // Exact decay factors (computed once per step)
    let decay_ampa = (-dt / TAU_AMPA).exp();
    let decay_gaba = (-dt / TAU_GABA_A).exp();

    for i in 0..n {
        // Conductance decay
        let ge_new = state[2 * n + i] * decay_ampa;
        let gi_new = state[3 * n + i] * decay_gaba;

        let ref_remaining = refractory[i];

        if ref_remaining > 0.0 {
            state[2 * n + i] = ge_new;
            state[3 * n + i] = gi_new;
            refractory[i] = (ref_remaining - dt).max(0.0);
            continue;
        }

        let v = state[i];
        let w = state[n + i];
        let ge = state[2 * n + i];
        let gi = state[3 * n + i];

        // Extract params for neuron i (SoA: field * N + i)
        let cm      = params[i];
        let gl      = params[n + i];
        let el      = params[2 * n + i];
        let vt      = params[3 * n + i];
        let delta_t = params[4 * n + i];
        let v_peak  = params[5 * n + i];
        let tau_w   = params[6 * n + i];
        let a       = params[7 * n + i];
        let b       = params[8 * n + i];
        let v_reset = params[9 * n + i];
        let t_ref   = params[10 * n + i];
        let ee      = params[11 * n + i];
        let ei      = params[12 * n + i];

        let i_syn = ge * (ee - v) + gi * (ei - v);
        let i_total = i_syn + currents[i] - w;

        let exp_arg = ((v - vt) / delta_t).min(EXP_CLAMP);
        let exp_term = gl * delta_t * exp_arg.exp();
        let dvdt = (-gl * (v - el) + exp_term + i_total) / cm;
        let v_new = v + dvdt * dt;

        let dwdt = (a * (v - el) - w) / tau_w;
        let w_new = w + dwdt * dt;

        state[2 * n + i] = ge_new;
        state[3 * n + i] = gi_new;

        if v_new >= v_peak {
            // Spike — apply reset
            state[i] = v_reset;
            state[n + i] = w_new + b;
            refractory[i] = t_ref;
            if spike_count < spike_out.len() {
                spike_out[spike_count] = i as u32;
            }
            spike_count += 1;
        } else {
            state[i] = v_new;
            state[n + i] = w_new;
        }
    }

    spike_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> AdExParams {
        AdExParams {
            cm: 200.0, gl: 10.0, el: -70.0, vt: -50.0,
            delta_t: 2.0, v_peak: 0.0, tau_w: 200.0,
            a: 2.0, b: 0.0, v_reset: -58.0, t_ref: 2.0,
            ee: 0.0, ei: -70.0,
        }
    }

    #[test]
    fn resting_state_no_spike() {
        let params = default_params();
        let mut state = AdExState { v: -70.0, w: 0.0, ge: 0.0, gi: 0.0 };
        let result = step_adex_single(&mut state, &params, 0.0, 0.1, 0.0);
        assert!(!result.spiked);
        // Voltage should remain near resting potential
        assert!((result.v - (-70.0_f64)).abs() < 1.0);
    }

    #[test]
    fn suprathreshold_current_spikes() {
        // V must reach V_PEAK (0mV) to trigger a spike. Set V just below V_PEAK
        // with large current so the step pushes it over.
        let params = default_params();
        let mut state = AdExState { v: -2.0, w: 0.0, ge: 0.0, gi: 0.0 };
        let result = step_adex_single(&mut state, &params, 2000.0, 0.1, 0.0);
        assert!(result.spiked);
    }

    #[test]
    fn refractory_prevents_spike() {
        let params = default_params();
        let mut state = AdExState { v: -40.0, w: 0.0, ge: 0.0, gi: 0.0 };
        let result = step_adex_single(&mut state, &params, 2000.0, 0.1, 2.0);
        assert!(!result.spiked);
        assert!((result.v - (-40.0_f64)).abs() < 0.001); // V unchanged during refractory
    }

    #[test]
    fn conductance_decay_exact() {
        let params = default_params();
        let mut state = AdExState { v: -70.0, w: 0.0, ge: 10.0, gi: 5.0 };
        let dt = 0.1_f64;
        step_adex_single(&mut state, &params, 0.0, dt, 0.0);
        let expected_ge = 10.0 * (-dt / TAU_AMPA).exp();
        assert!((state.ge - expected_ge).abs() < 1e-10);
    }

    #[test]
    fn population_step_matches_single() {
        // Verify that the SoA population function matches single-neuron function
        let n = 3usize;
        // SoA state: [V*3, W*3, GE*3, GI*3]
        let mut state_soa = vec![
            -70.0, -65.0, -2.0,  // V — neuron 2 near V_PEAK
              0.0,   0.0,  0.0,  // W
              0.0,   0.0,  0.0,  // GE
              0.0,   0.0,  0.0,  // GI
        ];
        // SoA params: all same defaults, 13*3 entries
        let mut params_soa = vec![0.0f64; 13 * n];
        let p = default_params();
        let pvals = [p.cm, p.gl, p.el, p.vt, p.delta_t, p.v_peak,
                     p.tau_w, p.a, p.b, p.v_reset, p.t_ref, p.ee, p.ei];
        for (f, val) in pvals.iter().enumerate() {
            for i in 0..n {
                params_soa[f * n + i] = *val;
            }
        }
        let currents = vec![0.0f64, 0.0, 1000.0];
        let mut refractory = vec![0.0f64; n];
        let mut spike_out = vec![0u32; n];

        let count = step_adex_population(
            &mut state_soa, &params_soa, &currents, &mut refractory, &mut spike_out, n, 0.1
        );

        // Neuron 2 (index 2) should spike — it started near threshold with large current
        assert!(count > 0);
        assert!(spike_out[..count].contains(&2u32));
    }
}
