"use client";

import { Card, Flex, Grid, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import { useScenarioStore } from "@/state/scenarioStore";

const summaryMetrics = [
  { label: "MAPE", value: "8.4%", delta: "-1.2% vs benchmark" },
  { label: "RMSE", value: "5.9", delta: "-0.7" },
  { label: "Correlation", value: "0.82", delta: "+0.05" },
];

export function AnalyticsPreview() {
  const status = useScenarioStore((state) => state.status);

  return (
    <Card className="h-full border border-slate-800 bg-slate-950/40">
      <Flex justifyContent="between" alignItems="center" className="mb-4">
        <div>
          <Title className="text-xl text-slate-100">Analytics Preview</Title>
          <Text className="text-slate-400">
            Headline metrics and diagnostics appear once a run completes.
          </Text>
        </div>
        <Text className="text-xs uppercase tracking-wide text-slate-500">{status === "completed" ? "Ready" : "Pending"}</Text>
      </Flex>
      <TabGroup>
        <TabList variant="line" className="bg-transparent text-slate-200">
          <Tab>Summary</Tab>
          <Tab>Salary tiers</Tab>
          <Tab>Diagnostics</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Grid numItems={1} className="gap-4 md:grid-cols-3">
              {summaryMetrics.map((metric) => (
                <Card key={metric.label} className="border border-slate-800 bg-slate-900/50 p-4">
                  <Title className="text-sm text-slate-400">{metric.label}</Title>
                  <Text className="text-2xl font-semibold text-slate-100">{metric.value}</Text>
                  <Text className="text-xs text-slate-500">{metric.delta}</Text>
                </Card>
              ))}
            </Grid>
          </TabPanel>
          <TabPanel>
            <div className="rounded-lg border border-dashed border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-500">
              Salary tier visuals will render here once backend APIs are connected.
            </div>
          </TabPanel>
          <TabPanel>
            <div className="rounded-lg border border-dashed border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-500">
              Residual and error diagnostics will appear here after run completion.
            </div>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </Card>
  );
}
