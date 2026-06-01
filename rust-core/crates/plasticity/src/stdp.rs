/*!
 * STDP kernel — Rust implementation for WASM acceleration.
 *
 * Nearest-neighbor STDP trace decay and weight update computations.
 * These match the TypeScript implementations in packages/plasticity/src/stdp.ts.
 */

/// Decay all traces in place: x(t+dt) = x(t) * exp(-dt/tau)
/// Applied to both pre-synaptic (per synapse) and post-synaptic (per target) trace arrays.
pub fn decay_traces_batch(traces: &mut [f32], decay_factor: f32) {
    for t in traces.iter_mut() {
        *t *= decay_factor;
    }
}

/// Apply STDP LTD when a pre-synaptic neuron fires.
/// Returns weight delta: ΔW = -a_minus * post_trace
#[inline]
pub fn apply_stdp_pre_spike(a_minus: f32, post_trace: f32) -> f32 {
    -a_minus * post_trace
}

/// Apply STDP LTP when a post-synaptic neuron fires.
/// Returns weight delta: ΔW = +a_plus * pre_trace
#[inline]
pub fn apply_stdp_post_spike(a_plus: f32, pre_trace: f32) -> f32 {
    a_plus * pre_trace
}

/// Batch STDP update for all synapses affected by pre-spike events.
///
/// For each source neuron in `pre_spike_sources`, updates weights
/// of all outgoing synapses using the post-synaptic traces.
pub fn batch_pre_spike_updates(
    pre_spike_sources: &[u32],
    row_ptr: &[i32],
    col_idx: &[i32],
    weights: &mut [f32],
    post_traces: &[f32],
    a_minus: f32,
    w_min: f32,
    w_max: f32,
) {
    for &src in pre_spike_sources {
        let src = src as usize;
        if src >= row_ptr.len().saturating_sub(1) { continue; }

        let start = row_ptr[src] as usize;
        let end   = row_ptr[src + 1] as usize;

        for idx in start..end {
            let tgt = col_idx[idx] as usize;
            if tgt < post_traces.len() && idx < weights.len() {
                let delta = apply_stdp_pre_spike(a_minus, post_traces[tgt]);
                weights[idx] = (weights[idx] + delta).clamp(w_min, w_max);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_decay_is_exact_exponential() {
        let dt = 0.1_f32;
        let tau = 20.0_f32;
        let decay = (-dt / tau).exp();
        let mut traces = vec![1.0f32, 0.5, 0.0];
        decay_traces_batch(&mut traces, decay);
        let expected = 1.0_f32 * decay;
        assert!((traces[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn stdp_ltd_on_pre_spike() {
        let delta = apply_stdp_pre_spike(0.0105, 0.5);
        assert!(delta < 0.0);
        assert!((delta - (-0.00525_f32)).abs() < 1e-6);
    }

    #[test]
    fn stdp_ltp_on_post_spike() {
        let delta = apply_stdp_post_spike(0.01, 0.8);
        assert!(delta > 0.0);
        assert!((delta - 0.008_f32).abs() < 1e-6);
    }
}
