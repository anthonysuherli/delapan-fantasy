import { AnalyticsPreview } from "@/components/run/AnalyticsPreview";
import { RunConsole } from "@/components/run/RunConsole";
import { PresetRail } from "@/components/layout/PresetRail";
import { ScenarioWizard } from "@/components/wizard/ScenarioWizard";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-8 lg:flex-row">
        <div className="lg:w-72">
          <PresetRail />
        </div>
        <div className="flex flex-1 flex-col gap-6">
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <ScenarioWizard />
            <RunConsole />
          </div>
          <AnalyticsPreview />
        </div>
      </div>
    </main>
  );
}
