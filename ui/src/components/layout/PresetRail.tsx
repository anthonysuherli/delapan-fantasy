"use client";

import { presetLibrary } from "@/lib/presets";
import { useScenarioStore } from "@/state/scenarioStore";
import { Card, Flex, Text, Title } from "@tremor/react";
import { CheckCircleIcon } from "@heroicons/react/24/solid";
import { clsx } from "clsx";

export function PresetRail() {
  const activePreset = useScenarioStore((state) => state.config.presetId);
  const selectPreset = useScenarioStore((state) => state.selectPreset);

  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto bg-slate-950/60 p-4">
      <div>
        <Title className="text-slate-100">Presets</Title>
        <Text className="text-slate-400">
          Start with a saved configuration or craft your own scenario from scratch.
        </Text>
      </div>
      <div className="space-y-3">
        {presetLibrary.map((preset) => {
          const isActive = preset.id === activePreset;
          return (
            <button
              key={preset.id}
              type="button"
              onClick={() => selectPreset(preset)}
              className="w-full text-left"
            >
              <Card
                className={clsx(
                  "border border-slate-800 bg-slate-900/60 transition hover:border-sky-500/60 hover:shadow-card",
                  isActive && "ring-2 ring-sky-400"
                )}
              >
                <Flex justifyContent="between" alignItems="start" className="gap-4">
                  <div>
                    <Title className="text-slate-100 text-lg">{preset.name}</Title>
                    <Text className="text-slate-400 text-sm">{preset.description}</Text>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-300">
                      <span className="rounded bg-slate-800/80 px-2 py-1">
                        Feature set: {preset.config.featureSet}
                      </span>
                      <span className="rounded bg-slate-800/80 px-2 py-1">
                        Model: {preset.config.model}
                      </span>
                      <span className="rounded bg-slate-800/80 px-2 py-1">
                        Window: {preset.config.window}
                      </span>
                    </div>
                  </div>
                  {isActive && <CheckCircleIcon className="h-6 w-6 text-sky-400" />}
                </Flex>
              </Card>
            </button>
          );
        })}
      </div>
    </aside>
  );
}
