"use client";

import { useEffect } from "react";
import { Badge, Button, Card, Flex, Text, Title } from "@tremor/react";
import { ArrowPathIcon } from "@heroicons/react/24/outline";
import { useScenarioStore } from "@/state/scenarioStore";

export function RunConsole() {
  const { status, logs, appendLog, setStatus } = useScenarioStore((state) => ({
    status: state.status,
    logs: state.logs,
    appendLog: state.appendLog,
    setStatus: state.setStatus,
  }));

  useEffect(() => {
    if (status !== "running") {
      return;
    }

    const timer = setTimeout(() => {
      appendLog("Validation complete. Aggregating metrics...");
      setStatus("completed");
      appendLog("Backtest completed successfully. Reports ready for download.");
    }, 4_000);

    return () => clearTimeout(timer);
  }, [status, appendLog, setStatus]);

  return (
    <Card className="flex h-full flex-col gap-4 border border-slate-800 bg-slate-950/40">
      <Flex justifyContent="between" alignItems="center">
        <div>
          <Title className="text-xl text-slate-100">Run Console</Title>
          <Text className="text-slate-400">Real-time feedback on queued jobs and worker activity.</Text>
        </div>
        <Badge color={status === "completed" ? "emerald" : status === "failed" ? "rose" : status === "running" ? "sky" : "slate"}>
          {status.toUpperCase()}
        </Badge>
      </Flex>
      <div className="flex items-center justify-end gap-2">
        <Button
          icon={ArrowPathIcon}
          variant="light"
          color="slate"
          onClick={() => appendLog("Manual refresh triggered")}
        >
          Refresh
        </Button>
      </div>
      <div className="grow overflow-hidden rounded-lg border border-slate-900 bg-black/60">
        <pre className="h-full overflow-y-auto p-4 text-xs leading-relaxed text-slate-300">
          {logs.length === 0
            ? "No logs yet. Launch a backtest to stream worker output here."
            : logs.join("\n")}
        </pre>
      </div>
    </Card>
  );
}
