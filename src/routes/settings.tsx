import { Page, PageHeader, Section } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { PathField } from "@/components/settings/path-field";
import { ThemePicker } from "@/components/settings/theme-picker";
import { backend } from "@/lib/api";
import { useEnvStatus } from "@/lib/api/queries";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { useThemeStore } from "@/stores/theme";

const BACKEND_NOTE = {
  tauri: null,
  http: "Connected to the development bridge.",
  fixture: "Showing sample data. Nothing here touches your files.",
} as const;

export function SettingsScreen() {
  const env = useEnvStatus();
  const { state: install, install: startInstall } = useEnvInstall();
  const { preference, setPreference } = useThemeStore();

  return (
    <Page>
      <PageHeader title="Settings" description={BACKEND_NOTE[backend.kind]} />

      <Section title="Environment" aside="One Python environment, managed by uv, runs every judge.">
        {env.error ? (
          <ErrorState
            title="The environment could not be inspected"
            error={env.error}
            onRetry={() => void env.refetch()}
          />
        ) : !env.data ? (
          <LoadingLines lines={5} />
        ) : (
          <EnvironmentPanel
            env={env.data}
            install={install}
            onInstall={(framework) => void startInstall(framework)}
          />
        )}
      </Section>

      <Section title="Appearance">
        <ThemePicker value={preference} onChange={setPreference} />
      </Section>

      <Section title="Content" aside="Problems, your solutions, and the Python project live here.">
        {env.data ? (
          <div className="flex max-w-measure flex-col gap-2">
            <PathField path={env.data.root} label="Content root" />
            <p className="text-xs text-muted-foreground">
              Solutions are plain files under <span className="selectable font-mono">workspace/</span>. Edit
              them in any editor; Whetstone re-reads the open file when its window regains focus.
            </p>
          </div>
        ) : (
          <LoadingLines lines={1} />
        )}
      </Section>
    </Page>
  );
}
