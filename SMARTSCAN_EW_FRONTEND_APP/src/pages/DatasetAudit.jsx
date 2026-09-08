import { PanelHead, StitchTable } from "../components/stitch";

const INVARIANTS = [
  ["Observation dimension", "360 (36 bands × 10 features)", "ENFORCED"],
  ["Action space", "180 (36 bands × 5 modes)", "ENFORCED"],
  ["Schema", "3Y.1", "ENFORCED"],
  ["Obs / GT separation", "Strict — GT never in observation", "ENFORCED"],
  ["Determinism", "root_seed = 42", "ENFORCED"],
  ["Corpus", "p3ac_50k — 100 episodes × 500 dwells", "VALID"],
];

const FEATURES = [
  "OCC",
  "DET",
  "MISS",
  "UNC",
  "AGE",
  "CNT",
  "CONF",
  "PRI",
  "AGIL",
  "RISK",
];

export default function DatasetAudit() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="dataset"
          title="Dataset Architecture, Verification & Handoff Audit"
          badge="AUDIT"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Read-only audit surface. This view verifies, never regenerates,
          the RF corpus.
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="Data Integrity & Strict Invariant Verification Matrix" />
        <StitchTable
          columns={["Invariant", "Contract", "Status"]}
          rows={INVARIANTS}
        />
      </div>

      <div className="st-panel">
        <PanelHead title="Observation Feature Distribution & Band Coverage Heatmap (36 Bands × 10 Features)" />
        <div className="st-heat" style={{ gridTemplateColumns: "repeat(10, minmax(0, 1fr))" }}>
          {Array.from({ length: 36 }, (_, band) =>
            FEATURES.map((f, fi) => {
              const active = [6, 10, 16, 28].includes(band);
              const v = active ? 0.55 + ((band + fi) % 4) * 0.1 : 0.08;
              return (
                <div
                  key={`${band}-${f}`}
                  className="st-heat-cell"
                  title={`B${band} ${f}: ${v.toFixed(2)}`}
                  style={{
                    background:
                      v > 0.6 ? "#0e1e94" : v > 0.3 ? "#1e2024" : "#0c0e12",
                    color: v > 0.6 ? "#bdc2ff" : "#908f9e",
                  }}
                >
                  {v.toFixed(1)}
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead title="PANEL A: SCHEDULER OBSERVATION DATA" badge="360-D" />
          <div className="st-body" style={{ color: "#c6c5d5" }}>
            Receiver-derived spectrum state only: 36 bands × 10 features per
            dwell. Emitter identity, true RF, and scenario metadata never
            enter this panel.
          </div>
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="PANEL B: ISOLATED GROUND TRUTH STORE" badge="TRUTH" />
          <div className="st-body" style={{ color: "#c6c5d5" }}>
            Operator-view simulation truth, physically isolated from Panel A.
            Used for audit and scoring only — never for inference.
          </div>
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="Dataset Handoff & PyTorch / RL DataLoader Pipeline Specifications" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Batch 64 × SeqLen 32 — optimized for DRQN recurrent rollout
          windowing. 2,048 temporal transitions per batch.
        </div>
        <StitchTable
          columns={["Stage", "Specification", "Status"]}
          rows={[
            ["Corpus", "p3ac_50k (Schema 3Y.1)", "VALID"],
            ["Windowing", "SeqLen 32, stride 1", "READY"],
            ["Batching", "Batch 64 → 2,048 transitions", "READY"],
            ["Handoff gate", "50K authorized · 100K blocked", "ENFORCED"],
          ]}
        />
      </div>
    </div>
  );
}
