// The shape of results/dashboard.json, written by Models/PHASE-Net/export_dashboard.py.
// Keep these in step with that script - it is the source of truth, this is only the view.

export type Num = number | null;

export interface Subject {
  id: string;
  held_out: boolean;
  hr_reference: Num;
  hr_predicted: Num;
  error: Num; // signed, for Bland-Altman
  abs_error: Num;
  mean_hr: Num; // Bland-Altman x axis
  snr_db: Num;
  macc: Num;
  duration_s: Num;
  waveform?: string;
  windowed?: {
    paired_mae: Num;
    median_model: Num;
    median_reference: Num;
    device_hr: Num;
    device_mae: Num;
    n_windows: number;
  };
}

export interface BlandAltman {
  bias_bpm: Num;
  sd_bpm: Num;
  loa_lower_bpm: Num;
  loa_upper_bpm: Num;
}

export interface Validation {
  dataset: Record<string, unknown>;
  model: Record<string, unknown>;
  preprocessing: Record<string, unknown>;
  hr_extraction: Record<string, unknown>;
  metrics_video_level: {
    n_subjects: number;
    mae_bpm: Num;
    rmse_bpm: Num;
    mape_pct: Num;
    pearson_r: Num;
    bland_altman: BlandAltman;
    within_3bpm_pct: Num;
    within_5bpm_pct: Num;
    mean_snr_db: Num;
    mean_macc: Num;
  };
  metrics_windowed: {
    protocol: Record<string, unknown>;
    held_out: { paired_mae: Num; device_mae: Num; n_subjects: number };
    all: { paired_mae: Num; device_mae: Num; n_subjects: number };
    total_windows: number;
  };
  subjects: Subject[];
  n_held_out: number;
}

export interface MethodScore {
  median_hr: Num;
  mae_vs_reference: Num;
  mae_vs_device: Num;
}

export interface Comparison {
  methods: string[];
  protocol: Record<string, unknown>;
  summary: Record<
    string,
    {
      held_out: Record<string, { mean: Num; worst: Num }>;
      all: Record<string, { mean: Num; worst: Num }>;
    }
  >;
  subjects: {
    id: string;
    held_out: boolean;
    device_hr: Num;
    methods: Record<string, MethodScore>;
  }[];
  n_held_out: number;
  n_all: number;
}

export interface MotionCondition {
  label: string;
  motion_dose: Num;
  displacement_max_px: Num;
  face_width_px: Num;
  duration_s: Num;
  methods: Record<
    string,
    {
      baseline_hr: Num;
      condition_hr: Num;
      drift: Num;
      still_sd: Num;
      condition_sd: Num;
      range: [Num, Num];
    }
  >;
}

export interface Hypothesis {
  hypothesis: string;
  test: string;
  result: string;
  evidence?: Record<string, Num>;
  note?: string;
}

export interface Motion {
  watch_bpm: Num;
  window_s: Num;
  conditions: MotionCondition[];
  noise_floor: Record<string, Num>;
  hypotheses_tested: Hypothesis[];
}

export interface ExtractionBenchmark {
  methods: {
    method: string;
    vs_reference: { mae_held_out: Num; mae_all: Num; worst: Num };
    vs_device: { mae_held_out: Num; mae_all: Num; reference_error: Num };
  }[];
  note: string;
}

export interface Dashboard {
  schema_version: string;
  generated_utc: string;
  project: {
    title: string;
    context: string;
    context_url: string;
    institution: string;
    repository: string;
  };
  headline: {
    held_out_mae_vs_reference: Num;
    held_out_mae_vs_device: Num;
    n_held_out_subjects: number;
    total_windows: number;
    webcam_noise_floor: Num;
    paper_claim: number;
  };
  limitations: string[];
  validation: Validation | null;
  comparison: Comparison | null;
  motion: Motion | null;
  extraction_benchmark: ExtractionBenchmark | null;
  traceability: {
    run: {
      id: string;
      created_utc: string;
      git_commit: string | null;
      environment: Record<string, unknown>;
      notes: string | null;
    } | null;
    model: Record<string, unknown> | null;
    note: string;
  };
}

export interface Waveform {
  subject: string;
  fs: number;
  n_samples: number;
  duration_s: number;
  signals: { predicted: number[]; reference: number[] };
  spectrum: { bpm: number[]; predicted: number[]; reference: number[] };
}
