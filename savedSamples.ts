// 공유 가능한 Samples 목록
// 이 파일은 커밋/푸시에 포함되어 모든 사용자가 공유할 수 있습니다.

import { ModuleType } from "./types";
import { SAMPLE_DATA } from "./sampleData";

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

// claim_data.csv 샘플 데이터 가져오기
const claimDataSample = SAMPLE_DATA.find((s) => s.name === "claim_data.csv");

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
          fileContent: claimDataSample?.content || "",
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
          inflation_rate: 5,
          amount_column: "클레임 금액",
          year_column: "연도",
        },
      },
      {
        type: ModuleType.SelectData,
        position: { x: 1100, y: 100 },
        name: "Select Data",
        parameters: {
          columnSelections: {
            "종목구분": {
              type: "string",
              selected: false,
            },
            "날짜": {
              type: "string",
              selected: false,
            },
            "연도": {
              type: "number",
              selected: true,
            },
            "클레임 금액": {
              type: "number",
              selected: false,
            },
            "기타": {
              type: "string",
              selected: false,
            },
            "클레임 금액_infl": {
              type: "number",
              selected: true,
            },
          },
        },
      },
      {
        type: ModuleType.SplitByThreshold,
        position: { x: 1396.2556393125353, y: 51.32331106295988 },
        name: "Split By Threshold",
        parameters: {
          threshold: 2500000,
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
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
          random_state: 43,
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
        parameters: {
          count_column: "count",
          selected_frequency_types: ["Poisson", "NegativeBinomial"],
        },
      },
      {
        type: ModuleType.FitSeverityModel,
        position: { x: 2000, y: 325 },
        name: "Fit Severity Model",
        parameters: {
          amount_column: "클레임 금액_infl",
          selected_severity_types: ["Lognormal", "Exponential", "Gamma", "Pareto"],
        },
      },
      {
        type: ModuleType.SimulateFreqServ,
        position: { x: 2300, y: 287.5 },
        name: "Simulate Freq-Sev Table",
        parameters: {
          frequency_type: "Poisson",
          severity_type: "Lognormal",
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
          simulation_count: 1000,
          custom_count: "",
          random_state: 43,
          output_format: "dfa",
        },
      },
      {
        type: ModuleType.CombineLossModel,
        position: { x: 2600, y: 175 },
        name: "Combine Loss Model",
        parameters: {},
      },
      {
        type: ModuleType.SettingThreshold,
        position: { x: 1251.4736165513566, y: -120.71432244354929 },
        name: "Setting Threshold 1",
        parameters: {
          target_column: "클레임 금액_infl",
          thresholds: [1000000, 2000000, 2500000],
          year_column: "연도",
        },
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
      {
        fromModuleIndex: 3,
        fromPort: "data_out",
        toModuleIndex: 12,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 12,
        fromPort: "threshold_out",
        toModuleIndex: 4,
        toPort: "threshold_in",
      },
    ],
  },
  {
    name: "XoL Loss Model",
    modules: [
      {
        type: ModuleType.LoadClaimData,
        position: { x: 100, y: 100 },
        name: "Load Claim Data",
        parameters: {
          source: "claim_data.csv",
          fileContent: claimDataSample?.content || "",
        },
      },
      {
        type: ModuleType.FormatChange,
        position: { x: 450, y: 100 },
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
          inflation_rate: 5,
          amount_column: "클레임 금액",
          year_column: "연도",
        },
      },
      {
        type: ModuleType.SelectData,
        position: { x: 1150, y: 190 },
        name: "Select Data",
        parameters: {
          columnSelections: {
            "연도": {
              type: "number",
              selected: true,
            },
            "클레임 금액_infl": {
              type: "number",
              selected: true,
            },
            "종목구분": {
              type: "string",
              selected: false,
            },
            "날짜": {
              type: "string",
              selected: false,
            },
            "클레임 금액": {
              type: "number",
              selected: false,
            },
            "기타": {
              type: "string",
              selected: false,
            },
          },
        },
      },
      {
        type: ModuleType.ApplyThreshold,
        position: { x: 1850, y: 280 },
        name: "Apply Threshold 1",
        parameters: {
          threshold: 100000,
          amount_column: "클레임 금액_infl",
          loss_column: "loss",
        },
      },
      {
        type: ModuleType.DefineXolContract,
        position: { x: 2900, y: 280 },
        name: "XoL Contract 1",
        parameters: {
          deductible: 3000000,
          limit: 2000000,
          reinstatements: 2,
          aggDeductible: 0,
          expenseRatio: 0.3,
          defaultReinstatementRate: 100,
          yearRates: [],
        },
      },
      {
        type: ModuleType.XolCalculator,
        position: { x: 3250, y: 640 },
        name: "XOL Calculator 1",
        parameters: {
          claim_column: "클레임 금액_infl",
        },
      },
      {
        type: ModuleType.XolPricing,
        position: { x: 3600, y: 640 },
        name: "XoL Pricing 1",
        parameters: {
          expenseRate: 0.2,
        },
      },
      {
        type: ModuleType.SplitByFreqServ,
        position: { x: 2200, y: 280 },
        name: "Split By Freq-Sev",
        parameters: {
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.FitFrequencyModel,
        position: { x: 2550, y: 280 },
        name: "Fit Frequency Model",
        parameters: {
          count_column: "count",
          selected_frequency_types: ["Poisson", "NegativeBinomial"],
        },
      },
      {
        type: ModuleType.FitSeverityModel,
        position: { x: 2550, y: 460 },
        name: "Fit Severity Model",
        parameters: {
          amount_column: "클레임 금액_infl",
          selected_severity_types: ["Lognormal", "Exponential", "Gamma", "Pareto"],
        },
      },
      {
        type: ModuleType.SimulateFreqServ,
        position: { x: 2900, y: 460 },
        name: "Simulate Freq-Sev Table",
        parameters: {
          frequency_type: "Poisson",
          severity_type: "Lognormal",
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
          simulation_count: 1000,
          custom_count: "",
          random_state: 43,
          output_format: "xol",
        },
      },
      {
        type: ModuleType.SettingThreshold,
        position: { x: 1500, y: 100 },
        name: "Setting Threshold 1",
        parameters: {
          target_column: null,
          thresholds: [],
          year_column: "",
        },
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
        fromModuleIndex: 5,
        fromPort: "contract_out",
        toModuleIndex: 6,
        toPort: "contract_in",
      },
      {
        fromModuleIndex: 6,
        fromPort: "data_out",
        toModuleIndex: 7,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 8,
        fromPort: "frequency_out",
        toModuleIndex: 9,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 8,
        fromPort: "severity_out",
        toModuleIndex: 10,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 9,
        fromPort: "data_out",
        toModuleIndex: 11,
        toPort: "frequency_in",
      },
      {
        fromModuleIndex: 10,
        fromPort: "data_out",
        toModuleIndex: 11,
        toPort: "severity_in",
      },
      {
        fromModuleIndex: 4,
        fromPort: "data_out",
        toModuleIndex: 8,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 11,
        fromPort: "output_2",
        toModuleIndex: 6,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 3,
        fromPort: "data_out",
        toModuleIndex: 12,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 12,
        fromPort: "threshold_out",
        toModuleIndex: 4,
        toPort: "threshold_in",
      },
    ],
  },
  {
    name: "XoL Experience",
    modules: [
      {
        type: ModuleType.LoadClaimData,
        position: { x: 200, y: 380 },
        name: "Load Claim Data",
        parameters: {
          source: "claim_data.csv",
          fileContent: claimDataSample?.content || "",
        },
      },
      {
        type: ModuleType.FormatChange,
        position: { x: 500, y: 380 },
        name: "Format Change",
        parameters: {
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.ApplyInflation,
        position: { x: 800, y: 380 },
        name: "Apply Inflation",
        parameters: {
          target_year: 2026,
          inflation_rate: 5,
          amount_column: "클레임 금액",
          year_column: "연도",
        },
      },
      {
        type: ModuleType.SelectData,
        position: { x: 1100, y: 380 },
        name: "Select Data",
        parameters: {
          columnSelections: {
            "연도": {
              type: "number",
              selected: true,
            },
            "클레임 금액_infl": {
              type: "number",
              selected: true,
            },
          },
        },
      },
      {
        type: ModuleType.ApplyThreshold,
        position: { x: 1397, y: 382 },
        name: "Apply Threshold 1",
        parameters: {
          threshold: 100000,
          amount_column: "클레임 금액_infl",
          loss_column: "loss",
        },
      },
      {
        type: ModuleType.DefineXolContract,
        position: { x: 1650, y: 237 },
        name: "XoL Contract 1",
        parameters: {
          deductible: 3000000,
          limit: 2000000,
          reinstatements: 2,
          aggDeductible: 0,
          expenseRatio: 0.3,
          defaultReinstatementRate: 100,
          yearRates: [],
        },
      },
      {
        type: ModuleType.XolCalculator,
        position: { x: 1685, y: 423 },
        name: "XOL Calculator 1",
        parameters: {
          claim_column: "클레임 금액_infl",
        },
      },
      {
        type: ModuleType.XolPricing,
        position: { x: 1982, y: 515 },
        name: "XoL Pricing 1",
        parameters: {
          expenseRate: 0.2,
        },
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
        fromModuleIndex: 5,
        fromPort: "contract_out",
        toModuleIndex: 6,
        toPort: "contract_in",
      },
      {
        fromModuleIndex: 4,
        fromPort: "data_out",
        toModuleIndex: 6,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 6,
        fromPort: "data_out",
        toModuleIndex: 7,
        toPort: "data_in",
      },
    ],
  },
];
