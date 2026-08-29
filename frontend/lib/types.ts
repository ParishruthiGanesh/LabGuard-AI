/** Mirrors `backend/labguard/models/domain.py`. */

export type AutonomyMode = "observe_only" | "safe_repair" | "managed_autonomy";

export type ClaimState =
  | "created"
  | "analyzing"
  | "skeptic_review"
  | "planning"
  | "awaiting_approval"
  | "executing"
  | "auditing"
  | "verdict"
  | "halted_budget"
  | "halted_loop"
  | "halted_approval";

export type JobState =
  | "planned"
  | "awaiting_approval"
  | "queued"
  | "running"
  | "recovering"
  | "completed"
  | "failed"
  | "blocked_loop"
  | "rejected";

export type SubclaimStatus =
  | "untested"
  | "testing"
  | "supported"
  | "contradicted"
  | "inconclusive";

export type LoopholeStatus = "open" | "investigating" | "confirmed" | "refuted" | "unresolved";
export type EvidenceStance = "supports" | "contradicts" | "neutral";
export type HealthStatus = "healthy" | "warning" | "critical" | "recovered" | "unknown";

export type VerdictStatus =
  | "supported"
  | "provisionally_supported"
  | "fragile"
  | "inconclusive"
  | "not_sufficiently_supported"
  | "refuted";

export type AgentName =
  | "claim_analyst"
  | "scientific_skeptic"
  | "experiment_planner"
  | "run_manager"
  | "run_medic"
  | "evidence_auditor"
  | "verdict_agent"
  | "orchestrator";

export interface ModelConfig {
  name: string;
  family: string;
  epochs: number;
  learning_rate: number;
  hidden_units: number;
  batch_size: number;
  class_weight: string;
  objective: string;
  is_baseline: boolean;
  role: string;
  notes: string;
}

export interface DatasetInfo {
  name: string;
  n_samples: number;
  n_features: number;
  positive_rate: number;
  test_fraction: number;
  inject_train_test_overlap: number;
  domain_shift_strength: number;
}

export interface ExistingResult {
  model_name: string;
  metric: string;
  value: number;
  seed: number;
  checkpoint_selected_on: string;
  epochs_trained: number;
  checkpoint_uri: string;
}

export interface BudgetPolicy {
  total_units: number;
  consumed_units: number;
  approval_threshold_units: number;
}

export interface ClaimContext {
  dataset: DatasetInfo;
  models: ModelConfig[];
  existing_results: ExistingResult[];
  reported_checkpoint_corrupt: boolean;
  notes: string;
}

export interface Claim {
  id: string;
  text: string;
  context: ClaimContext;
  autonomy_mode: AutonomyMode;
  budget: BudgetPolicy;
  state: ClaimState;
  active_agent: AgentName | null;
  latest_action: string;
  demo_mode: boolean;
  reasoning_backend: string;
  planning_round: number;
  halt_reason: string;
  created_at: string;
  updated_at: string;
}

export interface Subclaim {
  id: string;
  claim_id: string;
  statement: string;
  measurable_quantity: string;
  rationale: string;
  status: SubclaimStatus;
  confidence: number;
  evidence_ids: string[];
}

export interface Loophole {
  id: string;
  kind: string;
  title: string;
  rationale: string;
  severity: number;
  status: LoopholeStatus;
  detected_by: string;
  subclaim_ids: string[];
  resolution: string;
}

export interface AlternativeExplanation {
  id: string;
  statement: string;
  tested_by_action: string;
  status: string;
}

export interface PlanItem {
  id: string;
  action_type: string;
  params: Record<string, unknown>;
  reason: string;
  targets_loophole_ids: string[];
  targets_subclaim_ids: string[];
  estimated_cost_units: number;
  expected_information_gain: number;
  requires_approval: boolean;
  category: string;
  job_id: string | null;
}

export interface ExperimentPlan {
  id: string;
  claim_id: string;
  round_index: number;
  items: PlanItem[];
  summary: string;
  status: string;
  total_cost_units: number;
  requires_approval: boolean;
  approved_by: string;
  decided_at: string | null;
}

export interface EpochRecord {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_metric: number;
  val_metric: number;
  seconds: number;
  gpu_util_pct: number;
  memory_mb: number;
}

export interface HealthEvent {
  id: string;
  job_id: string;
  anomaly: string;
  status: HealthStatus;
  detail: string;
  epoch: number | null;
  action_taken: string;
  repaired: boolean;
  requires_approval: boolean;
  at: string;
}

export interface RunHealth {
  status: HealthStatus;
  summary: string;
  events: HealthEvent[];
  peak_memory_mb: number;
  mean_gpu_util_pct: number;
}

export interface Job {
  id: string;
  claim_id: string;
  plan_id: string;
  action_type: string;
  params: Record<string, unknown>;
  state: JobState;
  category: string;
  reason: string;
  attempts: number;
  max_retries: number;
  recovery_actions: string[];
  estimated_cost_units: number;
  actual_cost_units: number;
  error: string;
  fingerprint: string;
  curves: EpochRecord[];
  health: RunHealth;
  result: Record<string, unknown>;
  artifact_uris: string[];
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
}

export interface Evidence {
  id: string;
  job_id: string;
  stance: EvidenceStance;
  statement: string;
  measurements: Record<string, unknown>;
  strength: number;
  artifact_uris: string[];
  created_at: string;
}

export interface LedgerEntry {
  id: string;
  sequence: number;
  agent: AgentName;
  action: string;
  reason: string;
  input_summary: Record<string, unknown>;
  result_summary: Record<string, unknown>;
  decision: string;
  job_id: string | null;
  artifact_uris: string[];
  at: string;
}

export interface ScoreCheck {
  id: string;
  label: string;
  passed: boolean | null;
  weight: number;
  detail: string;
  observed: Record<string, unknown>;
}

export interface DimensionScore {
  dimension: string;
  score: number;
  checks: ScoreCheck[];
  calculation: string;
}

export interface ReliabilityScore {
  dimensions: DimensionScore[];
  overall: number;
  calculation: string;
}

export interface Verdict {
  id: string;
  status: VerdictStatus;
  headline: string;
  narrative: string;
  evidence_summary: string[];
  remaining_uncertainty: string[];
  run_health_incidents: string[];
  reproducibility: Record<string, unknown>;
  score: ReliabilityScore | null;
  rule_based_status: VerdictStatus;
  generated_by: string;
}

export interface ClaimSnapshot {
  claim: Claim;
  subclaims: Subclaim[];
  loopholes: Loophole[];
  alternatives: AlternativeExplanation[];
  plans: ExperimentPlan[];
  jobs: Job[];
  evidence: Evidence[];
  ledger: LedgerEntry[];
  verdict: Verdict | null;
  score: ReliabilityScore | null;
  revision: number;
  infrastructure: Record<string, string>;
  report_available: boolean;
}

export interface ActionSpecView {
  name: string;
  category: string;
  summary: string;
  base_cost_units: number;
  max_retries: number;
  min_autonomy: string;
  addresses: string[];
  parameters: Record<string, unknown>;
}

export interface AppConfig {
  infrastructure: Record<string, string>;
  autonomy_modes: string[];
  actions: ActionSpecView[];
  demo_scenario: {
    text: string;
    context: ClaimContext;
    budget: BudgetPolicy;
    autonomy_mode: AutonomyMode;
  };
}
