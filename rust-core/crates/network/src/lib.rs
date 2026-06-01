/*!
 * SNN network structures — Rust implementation.
 *
 * CSR connectivity operations optimized for spike propagation.
 * Used from WASM to compute synaptic current delivery.
 */

pub mod csr;

pub use csr::CSRSynapseMatrix;
