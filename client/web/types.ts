export interface ModelInfo {
  id: string
  name: string
  available: boolean
  hasResults: boolean
}

export interface CaptureResult {
  captureId: string
  imagePath: string
  width: number
  height: number
  timings: { captureMs: number }
}

export interface PredictionResult {
  predictionId: string
  raw: string
  x: number
  y: number
  resizedWh: [number, number]
  imagePath: string
  visualTokens: number
  attention: 'sdpa'
  timings: { inferenceMs: number }
}

export interface MetricValue {
  hits: number
  samples: number
  accuracy_percent: number
  parse_failures?: number
}

export interface BenchmarkMetrics {
  hits: number
  samples: number
  accuracy_percent: number
  parse_failures?: number
  runtime_seconds?: number
  by_official_group?: Record<string, MetricValue>
  by_ui_type?: Record<string, MetricValue>
  by_platform?: Record<string, MetricValue>
  by_app?: Record<string, MetricValue>
  by_application?: Record<string, MetricValue>
}

export interface MetricsFile {
  model: string
  base_model?: string
  model_path?: string
  metric?: string
  screenspot_pro: BenchmarkMetrics
  screenspot_v2: BenchmarkMetrics
}

export interface ResultEntry {
  modelId: string
  modelName: string
  path: string
  visualTokens: number | null
  metrics: MetricsFile | null
  error: string | null
}

export interface ReAGroundApi {
  getModels(): Promise<{ models: ModelInfo[] }>
  getResults(): Promise<{ results: ResultEntry[] }>
  capture(): Promise<CaptureResult>
  predict(payload: { modelId: string; captureId: string; instruction: string; visualTokens: number }): Promise<PredictionResult>
  cancel(predictionId: string): Promise<{ cancelled: boolean }>
  click(predictionId: string): Promise<{ clicked: boolean }>
}

declare global {
  interface Window {
    reaground: ReAGroundApi
  }
}

export {}
