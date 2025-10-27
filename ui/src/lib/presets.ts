import type { ScenarioConfig } from "@/state/scenarioStore";

export type Preset = {
  id: string;
  name: string;
  description: string;
  config: {
    featureSet: ScenarioConfig["modeling"]["featureSet"];
    model: ScenarioConfig["modeling"]["model"];
    window: string;
    defaults?: Partial<ScenarioConfig>;
  };
};

const makeWindowLabel = ({ trainStart, trainEnd, evalStart, evalEnd }: ScenarioConfig["dateRange"]): string => {
  return `${trainStart} → ${evalEnd}`;
};

const baseDateRange: ScenarioConfig["dateRange"] = {
  trainStart: "2023-10-01",
  trainEnd: "2024-02-15",
  evalStart: "2024-02-16",
  evalEnd: "2024-04-18",
};

export const presetLibrary: Preset[] = [
  {
    id: "balanced-core",
    name: "Balanced Core",
    description: "Core feature stack with XGBoost tuned for balanced risk across salary tiers.",
    config: {
      featureSet: "core_features_v1",
      model: "xgboost_regressor",
      window: makeWindowLabel(baseDateRange),
      defaults: {
        dateRange: baseDateRange,
        modeling: {
          hyperparameters: JSON.stringify(
            { learning_rate: 0.1, max_depth: 6, n_estimators: 800 },
            null,
            2
          ),
          perPlayer: true,
          recalibrationWeeks: 2,
        },
        filters: {
          includeInjured: false,
          minMinutes: 10,
          salaryTiers: [3500, 6000, 9500],
          savePredictions: true,
          saveModels: false,
        },
      },
    },
  },
  {
    id: "aggressive-ceiling",
    name: "Aggressive Ceiling",
    description: "Upside-focused blend using gradient boosting with looser injury filters.",
    config: {
      featureSet: "ceiling_signals_v2",
      model: "lightgbm_regressor",
      window: makeWindowLabel({
        trainStart: "2023-11-01",
        trainEnd: "2024-03-01",
        evalStart: "2024-03-02",
        evalEnd: "2024-04-18",
      }),
      defaults: {
        dateRange: {
          trainStart: "2023-11-01",
          trainEnd: "2024-03-01",
          evalStart: "2024-03-02",
          evalEnd: "2024-04-18",
        },
        modeling: {
          hyperparameters: JSON.stringify(
            { learning_rate: 0.07, num_leaves: 48, n_estimators: 900 },
            null,
            2
          ),
          perPlayer: true,
          recalibrationWeeks: 1,
        },
        filters: {
          includeInjured: true,
          minMinutes: 8,
          salaryTiers: [3200, 5800, 9200],
          savePredictions: true,
          saveModels: true,
        },
      },
    },
  },
  {
    id: "value-grind",
    name: "Value Grind",
    description: "Target value plays with ridge regression and tighter salary buckets.",
    config: {
      featureSet: "value_stream_v1",
      model: "ridge_regressor",
      window: makeWindowLabel({
        trainStart: "2023-10-15",
        trainEnd: "2024-02-01",
        evalStart: "2024-02-02",
        evalEnd: "2024-03-30",
      }),
      defaults: {
        dateRange: {
          trainStart: "2023-10-15",
          trainEnd: "2024-02-01",
          evalStart: "2024-02-02",
          evalEnd: "2024-03-30",
        },
        modeling: {
          hyperparameters: JSON.stringify({ alpha: 0.75 }, null, 2),
          perPlayer: false,
          recalibrationWeeks: 4,
        },
        filters: {
          includeInjured: false,
          minMinutes: 14,
          salaryTiers: [3000, 5200, 8200],
          savePredictions: false,
          saveModels: false,
        },
      },
    },
  },
];
