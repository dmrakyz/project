/*!
 * Plasticity kernels — Rust implementation.
 *
 * STDP trace decay and weight update kernels for WASM acceleration.
 * V2 will expose these via wasm-bindgen for use in the plasticity engine.
 */

pub mod stdp;
pub use stdp::{decay_traces_batch, apply_stdp_pre_spike, apply_stdp_post_spike};
