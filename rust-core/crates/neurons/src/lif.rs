/*!
 * Leaky Integrate-and-Fire (LIF) — Rust implementation.
 *
 * Simpler than AdEx. Used for large input populations and baseline comparisons.
 *
 * State SoA layout (3*N):
 *   state[0..N] = V (membrane voltage, mV)
 *   state[N..2N] = GE (excitatory conductance, nS)
 *   state[2N..3N] = GI (inhibitory conductance, nS)
 *
 * Params SoA layout (8*N):
 *   CM:0, GL:1, EL:2, VT:3, V_RESET:4, T_REF:5, EE:6, EI:7
 */

const TAU_AMPA: f64 = 5.0;
const TAU_GABA_A: f64 = 10.0;

pub struct LIFState {
    pub v: f64,
    pub ge: f64,
    pub gi: f64,
}

pub fn step_lif_population(
    state: &mut [f64],
    params: &[f64],
    currents: &[f64],
    refractory: &mut [f64],
    spike_out: &mut [u32],
    n: usize,
    dt: f64,
) -> usize {
    debug_assert_eq!(state.len(), 3 * n);
    debug_assert_eq!(params.len(), 8 * n);

    let mut spike_count = 0usize;
    let decay_ampa = (-dt / TAU_AMPA).exp();
    let decay_gaba = (-dt / TAU_GABA_A).exp();

    for i in 0..n {
        let ge_new = state[n + i] * decay_ampa;
        let gi_new = state[2 * n + i] * decay_gaba;

        state[n + i] = ge_new;
        state[2 * n + i] = gi_new;

        let ref_remaining = refractory[i];
        if ref_remaining > 0.0 {
            refractory[i] = (ref_remaining - dt).max(0.0);
            continue;
        }

        let v = state[i];
        let cm = params[i];
        let gl = params[n + i];
        let el = params[2 * n + i];
        let vt = params[3 * n + i];
        let v_reset = params[4 * n + i];
        let t_ref = params[5 * n + i];
        let ee = params[6 * n + i];
        let ei = params[7 * n + i];

        let i_syn = ge_new * (ee - v) + gi_new * (ei - v);
        let dvdt = (-gl * (v - el) + i_syn + currents[i]) / cm;
        let v_new = v + dvdt * dt;

        if v_new >= vt {
            state[i] = v_reset;
            refractory[i] = t_ref;
            if spike_count < spike_out.len() {
                spike_out[spike_count] = i as u32;
            }
            spike_count += 1;
        } else {
            state[i] = v_new;
        }
    }

    spike_count
}
