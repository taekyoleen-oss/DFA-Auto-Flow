// 공유 가능한 Samples 목록
// 이 파일은 커밋/푸시에 포함되어 모든 사용자가 공유할 수 있습니다.

import { ModuleType } from "./types";

export interface SavedSample {
  name: string;
  modules: Array<{
    type: ModuleType;
    position: { x: number; y: number };
    name: string;
    parameters?: Record<string, any>;
  }>;
  connections: Array<{
    fromModuleIndex: number;
    fromPort: string;
    toModuleIndex: number;
    toPort: string;
  }>;
}

export const SAVED_SAMPLES: SavedSample[] = [
  {
    name: "DFA Model",
    modules: [
      {
        type: ModuleType.LoadClaimData,
        position: { x: 200, y: 100 },
        name: "Load Claim Data",
        parameters: {
          source: "claim_data.csv",
          fileContent: "",
        },
      },
      {
        type: ModuleType.FormatChange,
        position: { x: 500, y: 100 },
        name: "Format Change",
        parameters: {
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.ApplyInflation,
        position: { x: 800, y: 100 },
        name: "Apply Inflation",
        parameters: {
          target_year: 2026,
          inflation_rate: 5.0,
          amount_column: "클레임 금액",
          year_column: "연도",
        },
      },
      {
        type: ModuleType.SelectData,
        position: { x: 1100, y: 100 },
        name: "Select Data",
        parameters: {
          columnSelections: {},
        },
      },
      {
        type: ModuleType.SplitByThreshold,
        position: { x: 1400, y: 100 },
        name: "Split By Threshold",
        parameters: {
          threshold: 1000000,
          amount_column: "클레임 금액_infl",
        },
      },
      {
        type: ModuleType.FitAggregateModel,
        position: { x: 1700, y: 25 },
        name: "Fit Agg Model",
        parameters: {
          selected_distributions: ["Lognormal", "Exponential", "Gamma", "Pareto"],
          amount_column: "클레임 금액_infl",
        },
      },
      {
        type: ModuleType.SimulateAggDist,
        position: { x: 2000, y: 25 },
        name: "Simulate Agg Table",
        parameters: {
          simulation_count: 10000,
          custom_count: "",
        },
      },
      {
        type: ModuleType.SplitByFreqServ,
        position: { x: 1700, y: 250 },
        name: "Split By Freq-Sev",
        parameters: {
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.FitFrequencyModel,
        position: { x: 2000, y: 250 },
        name: "Fit Frequency Model",
        parameters: {},
      },
      {
        type: ModuleType.FitSeverityModel,
        position: { x: 2000, y: 400 },
        name: "Fit Severity Model",
        parameters: {
          amount_column: "클레임 금액_infl",
        },
      },
      {
        type: ModuleType.SimulateFreqServ,
        position: { x: 2300, y: 325 },
        name: "Simulate Freq-Sev Table",
        parameters: {
          frequency_type: "Poisson",
          severity_type: "Lognormal",
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.CombineLossModel,
        position: { x: 2600, y: 175 },
        name: "Combine Loss Model",
        parameters: {},
      },
    ],
    connections: [
      {
        fromModuleIndex: 0,
        fromPort: "data_out",
        toModuleIndex: 1,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 1,
        fromPort: "data_out",
        toModuleIndex: 2,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 2,
        fromPort: "data_out",
        toModuleIndex: 3,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 3,
        fromPort: "data_out",
        toModuleIndex: 4,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 4,
        fromPort: "below_threshold_out",
        toModuleIndex: 5,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 5,
        fromPort: "model_out",
        toModuleIndex: 6,
        toPort: "model_in",
      },
      {
        fromModuleIndex: 4,
        fromPort: "above_threshold_out",
        toModuleIndex: 7,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 7,
        fromPort: "frequency_out",
        toModuleIndex: 8,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 7,
        fromPort: "severity_out",
        toModuleIndex: 9,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 8,
        fromPort: "data_out",
        toModuleIndex: 10,
        toPort: "frequency_in",
      },
      {
        fromModuleIndex: 9,
        fromPort: "data_out",
        toModuleIndex: 10,
        toPort: "severity_in",
      },
      {
        fromModuleIndex: 6,
        fromPort: "simulation_out",
        toModuleIndex: 11,
        toPort: "agg_dist_in",
      },
      {
        fromModuleIndex: 10,
        fromPort: "model_out",
        toModuleIndex: 11,
        toPort: "freq_serv_in",
      },
    ],
  },
];
