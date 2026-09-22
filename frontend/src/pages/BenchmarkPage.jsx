import { PanelHead } from '../components/stitch';
import BenchmarkTable from '../components/BenchmarkTable';

export default function BenchmarkPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="st-panel">
        <PanelHead
          icon="assessment"
          title="PROBLEM STATEMENT BENCHMARK EVALUATION"
          badge="7 FoM COMPARATIVE SUITE"
        />
        <div className="st-body" style={{ padding: 12 }}>
          <p style={{ margin: '0 0 12px 0', color: 'var(--text-muted)' }}>
            Empirical validation of the Cognitive SmartScan RL policy (DRQN + Stage 3 MoE) against standard open-loop scanning strategies (Random, RoundRobin, and HighestOccupancy) across 10 canonical TSRD scenarios over 5,000 receiver dwell steps.
          </p>
          <BenchmarkTable />
        </div>
      </div>
    </div>
  );
}
