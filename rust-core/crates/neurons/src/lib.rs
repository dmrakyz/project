/*!
 * SNN neuron model implementations in Rust — compiled to WebAssembly.
 *
 * This is the WASM core for V2+ performance. In V1, the TypeScript
 * reference implementation in packages/neurons/src/adex.ts is used instead.
 *
 * Integration tests in packages/neurons/src/adex.test.ts compare TypeScript
 * and Rust outputs on identical inputs to detect divergence.
 *
 * Architecture: No direct calls to JavaScript APIs from hot-path functions.
 * The WASM module receives state/params as flat f64 arrays (SoA layout),
 * performs integration, and returns spike booleans and updated state.
 *
 * SoA layout contract (must match TypeScript ADEX_STATE constants):
 *   state[0*N + i] = V (membrane voltage, mV)
 *   state[1*N + i] = W (adaptation current, pA)
 *   state[2*N + i] = GE (excitatory conductance, nS)
 *   state[3*N + i] = GI (inhibitory conductance, nS)
 *
 * SoA params layout (must match TypeScript ADEX_PARAMS constants):
 *   params[0*N + i] = CM (membrane capacitance, pF)
 *   params[1*N + i] = GL (leak conductance, nS)
 *   params[2*N + i] = EL (leak reversal, mV)
 *   params[3*N + i] = VT (spike threshold, mV)
 *   params[4*N + i] = DELTA_T (slope factor, mV)
 *   params[5*N + i] = V_PEAK (spike detection threshold, mV)
 *   params[6*N + i] = TAU_W (adaptation time constant, ms)
 *   params[7*N + i] = A (subthreshold adaptation, nS)
 *   params[8*N + i] = B (spike-triggered adaptation increment, pA)
 *   params[9*N + i] = V_RESET (reset potential, mV)
 *   params[10*N + i] = T_REF (refractory period, ms)
 *   params[11*N + i] = EE (excitatory reversal, mV)
 *   params[12*N + i] = EI (inhibitory reversal, mV)
 */

pub mod adex;
pub mod lif;

pub use adex::{step_adex_population, AdExState, AdExParams};
pub use lif::{step_lif_population, LIFState};
