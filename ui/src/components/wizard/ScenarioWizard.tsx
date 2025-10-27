"use client";

import { useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Divider,
  Flex,
  Grid,
  NumberInput,
  Switch,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  TextInput,
  Textarea,
  Title,
} from "@tremor/react";
import { ArrowLeftIcon, ArrowRightIcon, PlayIcon } from "@heroicons/react/24/outline";
import { ScenarioConfig, useScenarioStore } from "@/state/scenarioStore";

const stepLabels = [
  "Preset & Metadata",
  "Data Windows",
  "Modeling",
  "Filters",
  "Review",
];

export function ScenarioWizard() {
  const [activeStep, setActiveStep] = useState(0);
  const { config, updateConfig, setStatus, appendLog } = useScenarioStore((state) => ({
    config: state.config,
    updateConfig: state.updateConfig,
    setStatus: state.setStatus,
    appendLog: state.appendLog,
  }));

  const canAdvance = useMemo(() => {
    if (activeStep === 0) {
      return Boolean(config.runLabel.trim());
    }
    if (activeStep === 1) {
      return config.dateRange.trainStart < config.dateRange.trainEnd;
    }
    return true;
  }, [activeStep, config]);

  const onNext = () => setActiveStep((step) => Math.min(step + 1, stepLabels.length - 1));
  const onPrev = () => setActiveStep((step) => Math.max(step - 1, 0));

  const handleLaunch = () => {
    setStatus("queued");
    appendLog("Scenario queued. Awaiting worker acknowledgement...");
    setTimeout(() => {
      setStatus("running");
      appendLog("Worker picked up job. Streaming logs...");
    }, 500);
  };

  return (
    <Card className="h-full overflow-hidden border border-slate-800/80 bg-slate-950/40 p-0">
      <div className="border-b border-slate-800/70 bg-slate-900/80 px-8 py-6">
        <Flex justifyContent="between" alignItems="center">
          <div>
            <Title className="text-2xl text-slate-100">Scenario Builder</Title>
            <Text className="text-slate-400">
              Define your experiment parameters across data, modeling, and outputs.
            </Text>
          </div>
          <div className="flex items-center gap-2">
            {stepLabels.map((label, index) => (
              <Badge
                key={label}
                color={index === activeStep ? "sky" : "slate"}
                className="cursor-pointer"
                onClick={() => setActiveStep(index)}
              >
                {index + 1}
              </Badge>
            ))}
          </div>
        </Flex>
      </div>
      <div className="grid h-[calc(100%-5.5rem)] grid-cols-1 gap-0 md:grid-cols-[1.2fr_1fr]">
        <div className="h-full overflow-y-auto px-8 py-6">
          <TabGroup index={activeStep} onIndexChange={setActiveStep}>
            <TabList className="hidden">
              {stepLabels.map((label) => (
                <Tab key={label}>{label}</Tab>
              ))}
            </TabList>
            <TabPanels>
              <TabPanel>
                <StepPreset config={config} onUpdate={updateConfig} />
              </TabPanel>
              <TabPanel>
                <StepData config={config} onUpdate={updateConfig} />
              </TabPanel>
              <TabPanel>
                <StepModeling config={config} onUpdate={updateConfig} />
              </TabPanel>
              <TabPanel>
                <StepFilters config={config} onUpdate={updateConfig} />
              </TabPanel>
              <TabPanel>
                <StepReview config={config} onLaunch={handleLaunch} />
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </div>
        <div className="flex flex-col justify-between border-t border-slate-800/70 bg-slate-950/60 px-8 py-6 md:border-l md:border-t-0">
          <WizardStepSummary activeStep={activeStep} />
          <div className="mt-8 flex items-center justify-between gap-3">
            <Button
              icon={ArrowLeftIcon}
              variant="secondary"
              color="slate"
              onClick={onPrev}
              disabled={activeStep === 0}
            >
              Previous
            </Button>
            {activeStep < stepLabels.length - 1 ? (
              <Button
                icon={ArrowRightIcon}
                color="sky"
                disabled={!canAdvance}
                onClick={onNext}
              >
                Continue
              </Button>
            ) : (
              <Button icon={PlayIcon} color="emerald" onClick={handleLaunch}>
                Launch Backtest
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

type StepProps = {
  config: ScenarioConfig;
  onUpdate: (updater: (config: ScenarioConfig) => ScenarioConfig) => void;
};

function StepPreset({ config, onUpdate }: StepProps) {
  return (
    <div className="space-y-6">
      <div>
        <Title className="text-xl text-slate-100">Preset & Metadata</Title>
        <Text className="text-slate-400">
          Give the run a label, optionally refine the description, and confirm preset alignment.
        </Text>
      </div>
      <Grid numItems={1} className="gap-6 md:grid-cols-2">
        <TextInput
          value={config.runLabel}
          onChange={(event) =>
            onUpdate((draft) => ({ ...draft, runLabel: event.target.value }))
          }
          placeholder="Minutes recalibration sweep"
          className="bg-slate-900/60"
          aria-label="Run label"
        />
        <TextInput
          value={config.presetId ?? ""}
          onChange={(event) =>
            onUpdate((draft) => ({ ...draft, presetId: event.target.value || null }))
          }
          placeholder="Preset ID (optional)"
          className="bg-slate-900/60"
          aria-label="Preset id override"
        />
      </Grid>
      <Textarea
        value={config.description}
        onChange={(event) =>
          onUpdate((draft) => ({ ...draft, description: event.target.value }))
        }
        placeholder="Notes for future you."
        className="bg-slate-900/60"
        rows={4}
      />
      <Card className="border border-slate-800 bg-slate-900/60">
        <Title className="text-lg text-slate-100">Preset Snapshot</Title>
        <Text className="text-slate-400">
          Adjust the preset fields below to customize this run; the original preset remains unchanged.
        </Text>
        <Divider className="border-slate-800/70" />
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <Text className="text-slate-500 text-sm uppercase">Model</Text>
            <Text className="text-slate-200 text-base">{config.modeling.model}</Text>
          </div>
          <div>
            <Text className="text-slate-500 text-sm uppercase">Feature Set</Text>
            <Text className="text-slate-200 text-base">{config.modeling.featureSet}</Text>
          </div>
          <div>
            <Text className="text-slate-500 text-sm uppercase">Recalibration</Text>
            <Text className="text-slate-200 text-base">Every {config.modeling.recalibrationWeeks} weeks</Text>
          </div>
          <div>
            <Text className="text-slate-500 text-sm uppercase">Per Player Models</Text>
            <Text className="text-slate-200 text-base">{config.modeling.perPlayer ? "Enabled" : "Disabled"}</Text>
          </div>
        </div>
      </Card>
    </div>
  );
}

function StepData({ config, onUpdate }: StepProps) {
  return (
    <div className="space-y-6">
      <div>
        <Title className="text-xl text-slate-100">Data Windows</Title>
        <Text className="text-slate-400">
          Align training and evaluation windows and validate chronological ordering.
        </Text>
      </div>
      <Grid numItems={1} className="gap-4 md:grid-cols-2">
        <LabeledInput
          label="Train start"
          type="date"
          value={config.dateRange.trainStart}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dateRange: { ...draft.dateRange, trainStart: value },
            }))
          }
        />
        <LabeledInput
          label="Train end"
          type="date"
          value={config.dateRange.trainEnd}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dateRange: { ...draft.dateRange, trainEnd: value },
            }))
          }
        />
        <LabeledInput
          label="Evaluation start"
          type="date"
          value={config.dateRange.evalStart}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dateRange: { ...draft.dateRange, evalStart: value },
            }))
          }
        />
        <LabeledInput
          label="Evaluation end"
          type="date"
          value={config.dateRange.evalEnd}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dateRange: { ...draft.dateRange, evalEnd: value },
            }))
          }
        />
      </Grid>
      <Divider className="border-slate-800/70" />
      <Grid numItems={1} className="gap-4 md:grid-cols-2">
        <LabeledInput
          label="Schedule store"
          value={config.dataSources.schedulePath}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dataSources: { ...draft.dataSources, schedulePath: value },
            }))
          }
        />
        <LabeledInput
          label="Salary store"
          value={config.dataSources.salaryPath}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dataSources: { ...draft.dataSources, salaryPath: value },
            }))
          }
        />
      </Grid>
      <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
        <Text className="text-sm text-slate-300">Include official injury report feed</Text>
        <Switch
          checked={config.dataSources.injuryReport}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              dataSources: { ...draft.dataSources, injuryReport: value },
            }))
          }
        />
      </div>
    </div>
  );
}

function StepModeling({ config, onUpdate }: StepProps) {
  return (
    <div className="space-y-6">
      <div>
        <Title className="text-xl text-slate-100">Modeling</Title>
        <Text className="text-slate-400">
          Configure algorithms, feature bundles, and recalibration cadence.
        </Text>
      </div>
      <Grid numItems={1} className="gap-4 md:grid-cols-2">
        <LabeledInput
          label="Model registry id"
          value={config.modeling.model}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              modeling: { ...draft.modeling, model: value },
            }))
          }
        />
        <LabeledInput
          label="Feature bundle"
          value={config.modeling.featureSet}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              modeling: { ...draft.modeling, featureSet: value },
            }))
          }
        />
        <NumberInput
          className="bg-slate-900/60"
          min={1}
          step={1}
          value={config.modeling.recalibrationWeeks}
          onValueChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              modeling: {
                ...draft.modeling,
                recalibrationWeeks: parseNumericInput(
                  value,
                  draft.modeling.recalibrationWeeks
                ),
              },
            }))
          }
          placeholder="Recalibration cadence (weeks)"
        />
        <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
          <Text className="text-sm text-slate-300">Train per-player models</Text>
          <Switch
            checked={config.modeling.perPlayer}
            onChange={(value) =>
              onUpdate((draft) => ({
                ...draft,
                modeling: { ...draft.modeling, perPlayer: value },
              }))
            }
          />
        </div>
      </Grid>
      <Textarea
        className="bg-slate-900/60"
        rows={8}
        value={config.modeling.hyperparameters}
        onChange={(event) =>
          onUpdate((draft) => ({
            ...draft,
            modeling: { ...draft.modeling, hyperparameters: event.target.value },
          }))
        }
        aria-label="Hyperparameter JSON"
      />
    </div>
  );
}

function StepFilters({ config, onUpdate }: StepProps) {
  return (
    <div className="space-y-6">
      <div>
        <Title className="text-xl text-slate-100">Filters & Outputs</Title>
        <Text className="text-slate-400">
          Tailor cohort filters and decide which artifacts to persist.
        </Text>
      </div>
      <Grid numItems={1} className="gap-4 md:grid-cols-3">
        {config.filters.salaryTiers.map((tier, index) => (
          <NumberInput
            key={index}
            className="bg-slate-900/60"
            min={0}
            step={100}
            value={tier}
            onValueChange={(value) =>
              onUpdate((draft) => {
                const tiers = [...draft.filters.salaryTiers];
                tiers[index] = parseNumericInput(value, tiers[index]);
                return {
                  ...draft,
                  filters: { ...draft.filters, salaryTiers: tiers as [number, number, number] },
                };
              })
            }
            placeholder={`Tier ${index + 1}`}
          />
        ))}
      </Grid>
      <NumberInput
        className="bg-slate-900/60"
        min={0}
        step={1}
        value={config.filters.minMinutes}
        onValueChange={(value) =>
          onUpdate((draft) => ({
            ...draft,
            filters: {
              ...draft.filters,
              minMinutes: parseNumericInput(value, draft.filters.minMinutes),
            },
          }))
        }
        placeholder="Minimum minutes"
      />
      <div className="space-y-3">
        <ToggleRow
          label="Include players flagged as injured"
          checked={config.filters.includeInjured}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              filters: { ...draft.filters, includeInjured: value },
            }))
          }
        />
        <ToggleRow
          label="Persist predictions"
          checked={config.filters.savePredictions}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              filters: { ...draft.filters, savePredictions: value },
            }))
          }
        />
        <ToggleRow
          label="Persist trained models"
          checked={config.filters.saveModels}
          onChange={(value) =>
            onUpdate((draft) => ({
              ...draft,
              filters: { ...draft.filters, saveModels: value },
            }))
          }
        />
      </div>
    </div>
  );
}

type StepReviewProps = {
  config: ScenarioConfig;
  onLaunch: () => void;
};

function StepReview({ config, onLaunch }: StepReviewProps) {
  return (
    <div className="space-y-6">
      <div>
        <Title className="text-xl text-slate-100">Review</Title>
        <Text className="text-slate-400">
          Confirm the normalized configuration before submitting the backtest.
        </Text>
      </div>
      <Card className="border border-slate-800 bg-slate-900/60">
        <Title className="text-slate-100 text-lg">Summary</Title>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <SummaryItem label="Run label" value={config.runLabel || "Untitled scenario"} />
          <SummaryItem
            label="Preset"
            value={config.presetId ? config.presetId : "Custom"}
          />
          <SummaryItem
            label="Train window"
            value={`${config.dateRange.trainStart} → ${config.dateRange.trainEnd}`}
          />
          <SummaryItem
            label="Eval window"
            value={`${config.dateRange.evalStart} → ${config.dateRange.evalEnd}`}
          />
          <SummaryItem label="Model" value={config.modeling.model} />
          <SummaryItem label="Features" value={config.modeling.featureSet} />
        </div>
        <Divider className="border-slate-800/70" />
        <Textarea
          readOnly
          className="bg-slate-950/60 font-mono text-xs"
          rows={10}
          value={JSON.stringify(config, null, 2)}
        />
      </Card>
      <Button icon={PlayIcon} color="emerald" onClick={onLaunch}>
        Launch Backtest
      </Button>
    </div>
  );
}

function WizardStepSummary({ activeStep }: { activeStep: number }) {
  const status = useScenarioStore((state) => state.status);

  return (
    <div className="space-y-4">
      <div>
        <Title className="text-lg text-slate-100">Progress</Title>
        <Text className="text-slate-400">
          Step {activeStep + 1} of {stepLabels.length}
        </Text>
      </div>
      <div className="flex items-center gap-2">
        <Badge color="sky">{stepLabels[activeStep]}</Badge>
        <Badge color={status === "running" ? "emerald" : status === "queued" ? "amber" : "slate"}>
          {status.toUpperCase()}
        </Badge>
      </div>
      <Text className="text-sm text-slate-400">
        Use the right-hand controls to move between steps. Launching the backtest will transition the run into the monitoring console.
      </Text>
    </div>
  );
}

type LabeledInputProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
};

function LabeledInput({ label, value, onChange, type = "text" }: LabeledInputProps) {
  return (
    <label className="flex flex-col gap-2 text-sm text-slate-300">
      <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
      <TextInput
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="bg-slate-900/60"
      />
    </label>
  );
}

type ToggleRowProps = {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
};

function ToggleRow({ label, checked, onChange }: ToggleRowProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
      <Text className="text-sm text-slate-300">{label}</Text>
      <Switch checked={checked} onChange={onChange} />
    </div>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Text className="text-xs uppercase tracking-wide text-slate-500">{label}</Text>
      <Text className="text-slate-200 text-sm">{value}</Text>
    </div>
  );
}

function parseNumericInput(value: number | null | undefined, fallback: number): number {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return fallback;
  }

  return value;
}
