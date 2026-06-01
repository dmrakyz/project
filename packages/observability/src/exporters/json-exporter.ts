/**
 * JSON exporter for spike rasters and weight snapshots.
 * Produces files compatible with standard analysis tools.
 */

import type { SpikeRasterEntry, WeightSnapshot } from '@snn/types';

export interface ExportManifest {
  readonly version: string;
  readonly exportedAt: string;
  readonly simulationDt: number;
  readonly totalSimulatedMs: number;
}

export function exportSpikeRasterToJSON(
  spikes: SpikeRasterEntry[],
  manifest: Partial<ExportManifest> = {}
): string {
  return JSON.stringify({
    version: '1.0',
    exportedAt: new Date().toISOString(),
    ...manifest,
    spikes: spikes.map(s => ({ t: s.t, pop: s.populationId, n: s.neuronIndex })),
  }, null, 2);
}

export function exportWeightSnapshotsToJSON(snapshots: WeightSnapshot[]): string {
  return JSON.stringify({
    version: '1.0',
    exportedAt: new Date().toISOString(),
    snapshots: snapshots.map(s => ({
      t: s.t,
      projectionId: s.projectionId,
      mean: s.mean,
      std: s.std,
      min: s.min,
      max: s.max,
      // Weights stored as base64 for compactness
      weights: Buffer.from(s.weights.buffer).toString('base64'),
    })),
  }, null, 2);
}

export function exportSpikeRasterToCSV(spikes: SpikeRasterEntry[]): string {
  const header = 't_ms,population_id,neuron_index\n';
  const rows = spikes
    .map(s => `${s.t.toFixed(3)},${s.populationId},${s.neuronIndex}`)
    .join('\n');
  return header + rows;
}
