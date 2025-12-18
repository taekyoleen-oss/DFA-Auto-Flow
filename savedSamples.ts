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
  {
    name: "XoL Loss Model",
    modules: [
      {
        type: ModuleType.LoadClaimData,
        position: { x: 200, y: 100 },
        name: "Load Claim Data",
        parameters: {
          source: "claim_data.csv",
          fileContent: "종목구분,날짜,클레임 금액,기타\n자동차보험,2020-11-04,1085304,도난사고 - 251번 사고\n자동차보험,2020-04-05,653036,배상책임 - 90번 사고\n자동차보험,2020-10-14,1224631,교통사고 - 96번 사고\n건강보험,2020-04-08,2466779,기타사고 - 28번 사고\n건강보험,2020-09-07,604814,질병 - 226번 사고\n상해보험,2020-08-19,604822,교통사고 - 778번 사고\n배상책임보험,2020-03-23,2580181,상해사고 - 285번 사고\n상해보험,2020-03-07,1347750,화재사고 - 95번 사고\n상해보험,2020-07-04,501029,상해사고 - 619번 사고\n자동차보험,2020-01-15,987654,교통사고 - 123번 사고\n화재보험,2020-05-20,3456789,화재사고 - 456번 사고\n건강보험,2020-12-10,789012,질병 - 789번 사고\n자동차보험,2020-06-18,1123456,도난사고 - 321번 사고\n상해보험,2020-02-25,876543,낙상사고 - 654번 사고\n배상책임보험,2020-09-30,2345678,배상책임 - 987번 사고\n화재보험,2020-11-12,1567890,화재사고 - 147번 사고\n건강보험,2020-07-22,543210,질병 - 258번 사고\n자동차보험,2020-04-14,987654,교통사고 - 369번 사고\n상해보험,2020-10-05,765432,상해사고 - 741번 사고\n건강보험,2020-08-28,1234567,질병 - 852번 사고\n자동차보험,2021-03-15,1456789,교통사고 - 963번 사고\n화재보험,2021-07-22,2345678,화재사고 - 159번 사고\n상해보험,2021-11-08,987654,낙상사고 - 357번 사고\n건강보험,2021-05-30,1123456,질병 - 486번 사고\n배상책임보험,2021-09-14,3456789,배상책임 - 753번 사고\n자동차보험,2021-01-25,876543,도난사고 - 951번 사고\n상해보험,2021-12-18,1567890,상해사고 - 264번 사고\n화재보험,2021-04-07,543210,화재사고 - 582번 사고\n건강보험,2021-08-20,2345678,질병 - 693번 사고\n자동차보험,2021-06-12,765432,교통사고 - 147번 사고\n상해보험,2021-10-28,1234567,낙상사고 - 258번 사고\n배상책임보험,2021-02-14,987654,배상책임 - 369번 사고\n화재보험,2021-11-30,1456789,화재사고 - 741번 사고\n건강보험,2021-07-05,2345678,질병 - 852번 사고\n자동차보험,2021-09-22,876543,도난사고 - 963번 사고\n상해보험,2021-03-18,1123456,상해사고 - 159번 사고\n건강보험,2021-12-01,3456789,질병 - 357번 사고\n자동차보험,2021-05-14,543210,교통사고 - 486번 사고\n화재보험,2021-08-28,1567890,화재사고 - 753번 사고\n배상책임보험,2021-04-10,2345678,배상책임 - 951번 사고\n자동차보험,2022-01-20,987654,교통사고 - 264번 사고\n상해보험,2022-06-15,1456789,낙상사고 - 582번 사고\n건강보험,2022-10-08,2345678,질병 - 693번 사고\n화재보험,2022-03-25,876543,화재사고 - 147번 사고\n배상책임보험,2022-08-12,1123456,배상책임 - 258번 사고\n자동차보험,2022-12-28,3456789,도난사고 - 369번 사고\n상해보험,2022-05-18,543210,상해사고 - 741번 사고\n건강보험,2022-09-30,1567890,질병 - 852번 사고\n화재보험,2022-02-14,2345678,화재사고 - 963번 사고\n자동차보험,2022-11-22,987654,교통사고 - 159번 사고\n상해보험,2022-07-05,1456789,낙상사고 - 357번 사고\n배상책임보험,2022-04-28,2345678,배상책임 - 486번 사고\n건강보험,2022-08-18,876543,질병 - 753번 사고\n자동차보험,2022-12-10,1123456,도난사고 - 951번 사고\n화재보험,2022-06-25,3456789,화재사고 - 264번 사고\n상해보험,2022-10-14,543210,상해사고 - 582번 사고\n건강보험,2022-03-08,1567890,질병 - 693번 사고\n자동차보험,2022-09-20,2345678,교통사고 - 147번 사고\n배상책임보험,2022-05-12,987654,배상책임 - 258번 사고\n화재보험,2022-11-28,1456789,화재사고 - 369번 사고\n자동차보험,2023-02-15,2345678,교통사고 - 741번 사고\n상해보험,2023-07-22,876543,낙상사고 - 852번 사고\n건강보험,2023-11-08,1123456,질병 - 963번 사고\n화재보험,2023-04-30,3456789,화재사고 - 159번 사고\n배상책임보험,2023-09-14,543210,배상책임 - 357번 사고\n자동차보험,2023-01-25,1567890,도난사고 - 486번 사고\n상해보험,2023-06-18,2345678,상해사고 - 753번 사고\n건강보험,2023-10-05,987654,질병 - 951번 사고\n화재보험,2023-03-28,1456789,화재사고 - 264번 사고\n자동차보험,2023-08-12,2345678,교통사고 - 582번 사고\n배상책임보험,2023-12-20,876543,배상책임 - 693번 사고\n상해보험,2023-05-14,1123456,낙상사고 - 147번 사고\n건강보험,2023-09-30,3456789,질병 - 258번 사고\n화재보험,2023-02-08,543210,화재사고 - 369번 사고\n자동차보험,2023-11-18,1567890,도난사고 - 741번 사고\n상해보험,2023-07-25,2345678,상해사고 - 852번 사고\n건강보험,2023-04-10,987654,질병 - 963번 사고\n배상책임보험,2023-10-28,1456789,배상책임 - 159번 사고\n화재보험,2023-08-14,2345678,화재사고 - 357번 사고\n자동차보험,2024-01-20,876543,교통사고 - 486번 사고\n상해보험,2024-06-15,1123456,낙상사고 - 753번 사고\n건강보험,2024-10-08,3456789,질병 - 951번 사고\n화재보험,2024-03-25,543210,화재사고 - 264번 사고\n배상책임보험,2024-08-12,1567890,배상책임 - 582번 사고\n자동차보험,2024-12-28,2345678,도난사고 - 693번 사고\n상해보험,2024-05-18,987654,상해사고 - 147번 사고\n건강보험,2024-09-30,1456789,질병 - 258번 사고\n화재보험,2024-02-14,2345678,화재사고 - 369번 사고\n자동차보험,2024-11-22,876543,교통사고 - 741번 사고\n배상책임보험,2024-07-05,1123456,배상책임 - 852번 사고\n상해보험,2024-04-28,3456789,낙상사고 - 963번 사고\n건강보험,2024-08-18,543210,질병 - 159번 사고\n자동차보험,2024-12-10,1567890,도난사고 - 357번 사고\n화재보험,2024-06-25,2345678,화재사고 - 486번 사고\n상해보험,2024-10-14,987654,상해사고 - 753번 사고\n건강보험,2024-03-08,1456789,질병 - 951번 사고\n배상책임보험,2024-09-20,2345678,배상책임 - 264번 사고\n화재보험,2024-05-12,876543,화재사고 - 582번 사고",
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
        type: ModuleType.ApplyThreshold,
        position: { x: 1053.8332711112948, y: 89.32906767186675 },
        name: "Apply Threshold 1",
        parameters: {
          threshold: 1000000,
          amount_column: "클레임 금액",
        },
      },
      {
        type: ModuleType.SplitByFreqServ,
        position: { x: 1322.5357015202055, y: 86.71428647640563 },
        name: "Split By Freq-Sev",
        parameters: {
          amount_column: "클레임 금액_infl",
          date_column: "날짜",
        },
      },
      {
        type: ModuleType.FitFrequencyModel,
        position: { x: 1622.5357015202055, y: 11.714286476405633 },
        name: "Fit Frequency Model",
        parameters: {
          count_column: "count",
          selected_frequency_types: ["Poisson", "NegativeBinomial"],
        },
      },
      {
        type: ModuleType.FitSeverityModel,
        position: { x: 1622.5357015202055, y: 161.71428647640562 },
        name: "Fit Severity Model",
        parameters: {
          amount_column: "클레임 금액_infl",
          selected_severity_types: ["Lognormal", "Exponential", "Gamma", "Pareto"],
        },
      },
      {
        type: ModuleType.SimulateFreqServ,
        position: { x: 1922.5357015202055, y: 86.71428647640563 },
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
        type: ModuleType.DefineXolContract,
        position: { x: 2229.806008532251, y: -60.98878364404824 },
        name: "XoL Contract",
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
        position: { x: 2222.5357015202053, y: 161.71428647640562 },
        name: "XoL Calculator",
        parameters: {
          claim_column: "보험금",
          year_column: "시뮬레이션 번호",
        },
      },
      {
        type: ModuleType.XolPricing,
        position: { x: 2522.5357015202053, y: 161.71428647640562 },
        name: "XoL Pricing",
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
        fromModuleIndex: 4,
        fromPort: "frequency_out",
        toModuleIndex: 5,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 4,
        fromPort: "severity_out",
        toModuleIndex: 6,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 5,
        fromPort: "data_out",
        toModuleIndex: 7,
        toPort: "frequency_in",
      },
      {
        fromModuleIndex: 6,
        fromPort: "data_out",
        toModuleIndex: 7,
        toPort: "severity_in",
      },
      {
        fromModuleIndex: 7,
        fromPort: "output_2",
        toModuleIndex: 9,
        toPort: "data_in",
      },
      {
        fromModuleIndex: 8,
        fromPort: "contract_out",
        toModuleIndex: 9,
        toPort: "contract_in",
      },
      {
        fromModuleIndex: 9,
        fromPort: "data_out",
        toModuleIndex: 10,
        toPort: "data_in",
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
