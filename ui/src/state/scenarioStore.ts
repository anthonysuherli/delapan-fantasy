import { create } from "zustand";
import { Preset } from "@/lib/presets";

export type DateRange = {
  trainStart: string;
  trainEnd: string;
  evalStart: string;
  evalEnd: string;
};

export type ScenarioConfig = {
  presetId: string | null;
  runLabel: string;
  description: string;
  dateRange: DateRange;
  dataSources: {
    schedulePath: string;
    salaryPath: string;
    injuryReport: boolean;
  };
  modeling: {
    model: string;
    featureSet: string;
    hyperparameters: string;
    perPlayer: boolean;
    recalibrationWeeks: number;
  };
  filters: {
    includeInjured: boolean;
    minMinutes: number;
    salaryTiers: [number, number, number];
    savePredictions: boolean;
    saveModels: boolean;
  };
};

export type RunStatus = "idle" | "draft" | "validating" | "queued" | "running" | "completed" | "failed";

export type ScenarioState = {
  config: ScenarioConfig;
  status: RunStatus;
  logs: string[];
  selectPreset: (preset: Preset) => void;
  updateConfig: (updater: (config: ScenarioConfig) => ScenarioConfig) => void;
  appendLog: (line: string) => void;
  setStatus: (status: RunStatus) => void;
  reset: () => void;
};

const createDefaultConfig = (): ScenarioConfig => ({
  presetId: null,
  runLabel: "",
  description: "",
  dateRange: {
    trainStart: "2023-10-01",
    trainEnd: "2024-02-15",
    evalStart: "2024-02-16",
    evalEnd: "2024-04-18",
  },
  dataSources: {
    schedulePath: "storage://sqlite/nba_dfs.db",
    salaryPath: "s3://dfs-bucket/dk_salaries.parquet",
    injuryReport: true,
  },
  modeling: {
    model: "xgboost_regressor",
    featureSet: "core_features_v1",
    hyperparameters: JSON.stringify({ learning_rate: 0.1, max_depth: 6, n_estimators: 800 }, null, 2),
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
});

export const useScenarioStore = create<ScenarioState>((set) => ({
  config: createDefaultConfig(),
  status: "idle",
  logs: [],
  selectPreset: (preset) =>
    set((state) => {
      const defaults = preset.config.defaults ?? {};
      return {
        config: {
          ...state.config,
          presetId: preset.id,
          runLabel: preset.name,
          description: preset.description,
          dateRange: defaults.dateRange ?? state.config.dateRange,
          dataSources: {
            ...state.config.dataSources,
            ...(defaults.dataSources ?? {}),
          },
          modeling: {
            ...state.config.modeling,
            ...(defaults.modeling ?? {}),
            featureSet: preset.config.featureSet,
            model: preset.config.model,
          },
          filters: {
            ...state.config.filters,
            ...(defaults.filters ?? {}),
          },
        },
        status: "draft",
        logs: [
          ...state.logs,
          `[preset] ${preset.name} loaded with ${preset.config.featureSet} / ${preset.config.model}`,
        ],
      };
    }),
  updateConfig: (updater) =>
    set((state) => ({
      config: updater(state.config),
      status: state.status === "idle" ? "draft" : state.status,
    })),
  appendLog: (line) =>
    set((state) => ({
      logs: [...state.logs, `${new Date().toLocaleTimeString()} - ${line}`],
    })),
  setStatus: (status) => set({ status }),
  reset: () =>
    set({
      config: createDefaultConfig(),
      status: "idle",
      logs: [],
    }),
}));
