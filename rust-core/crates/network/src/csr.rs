/*!
 * CSR (Compressed Sparse Row) synapse matrix — Rust implementation.
 *
 * Provides efficient O(degree) access to outgoing synapses per source neuron.
 * This is the hot path during spike propagation: for each spike, scatter
 * conductance increments to all target neurons.
 *
 * Memory layout matches TypeScript CSRConnectivityStore exactly:
 *   row_ptr: [n_sources + 1] i32
 *   col_idx: [n_synapses]    i32
 *   weights: [n_synapses]    f32
 *   delays:  [n_synapses]    f32 (ms)
 */

pub struct CSRSynapseMatrix {
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    pub weights: Vec<f32>,
    pub delays: Vec<f32>,
    pub n_sources: usize,
    pub n_targets: usize,
}

impl CSRSynapseMatrix {
    pub fn new(
        row_ptr: Vec<i32>,
        col_idx: Vec<i32>,
        weights: Vec<f32>,
        delays: Vec<f32>,
        n_sources: usize,
        n_targets: usize,
    ) -> Self {
        assert_eq!(row_ptr.len(), n_sources + 1);
        assert_eq!(col_idx.len(), weights.len());
        assert_eq!(col_idx.len(), delays.len());
        Self { row_ptr, col_idx, weights, delays, n_sources, n_targets }
    }

    pub fn synapse_count(&self) -> usize {
        self.col_idx.len()
    }

    /// Deliver spikes from source neurons, incrementing target conductances.
    ///
    /// For each source neuron index in `spike_indices`, increments
    /// `target_conductances[target]` by `weights[synapse]` for all outgoing synapses.
    ///
    /// This is the spike propagation hot path.
    pub fn deliver_spikes(
        &self,
        spike_indices: &[u32],
        target_conductances: &mut [f32],
    ) {
        for &src in spike_indices {
            let src = src as usize;
            if src >= self.n_sources { continue; }

            let start = self.row_ptr[src] as usize;
            let end   = self.row_ptr[src + 1] as usize;

            for idx in start..end {
                let tgt = self.col_idx[idx] as usize;
                if tgt < target_conductances.len() {
                    target_conductances[tgt] += self.weights[idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deliver_spikes_increments_conductances() {
        // 2 sources, 3 targets
        // Source 0 connects to targets 0, 2 with weights 1.0, 0.5
        // Source 1 connects to target 1 with weight 2.0
        let row_ptr = vec![0i32, 2, 3];
        let col_idx = vec![0i32, 2, 1];
        let weights = vec![1.0f32, 0.5, 2.0];
        let delays  = vec![1.0f32, 1.0, 1.0];
        let matrix = CSRSynapseMatrix::new(row_ptr, col_idx, weights, delays, 2, 3);

        let mut conductances = vec![0.0f32; 3];
        matrix.deliver_spikes(&[0], &mut conductances);

        assert!((conductances[0] - 1.0).abs() < 1e-6);
        assert!((conductances[1] - 0.0).abs() < 1e-6);
        assert!((conductances[2] - 0.5).abs() < 1e-6);

        matrix.deliver_spikes(&[1], &mut conductances);
        assert!((conductances[1] - 2.0).abs() < 1e-6);
    }
}
