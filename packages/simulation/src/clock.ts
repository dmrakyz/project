/**
 * Simulation clock — the single source of truth for simulation time.
 *
 * All subsystems receive the current time from the clock.
 * No subsystem reads system time directly.
 * Simulation time is deterministic and can be paused, accelerated, or reset.
 *
 * The clock is the fundamental unit of computation in this system:
 * temporal dynamics are first-class, not an implementation detail.
 */

import type { SimulationClock } from '@snn/types';

export class SimClock implements SimulationClock {
  private _t: number = 0;
  private _step: number = 0;
  readonly dt: number;

  constructor(dt: number = 0.1) {
    if (dt <= 0) throw new Error(`Clock dt must be > 0, got ${dt}`);
    this.dt = dt;
  }

  get t(): number {
    return this._t;
  }

  get step(): number {
    return this._step;
  }

  tick(): void {
    this._step++;
    // Use step-based time to avoid floating-point accumulation errors
    this._t = this._step * this.dt;
  }

  reset(): void {
    this._t = 0;
    this._step = 0;
  }
}
