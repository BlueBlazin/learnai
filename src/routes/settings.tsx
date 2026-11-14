import { createLazyRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { Page, PageHeader, Section } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { ModelsPanel } from "@/components/models/models-panel";
import { ContentPanel } from "@/components/settings/content-panel";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { ThemePicker } from "@/components/settings/theme-picker";
import { backend } from "@/lib/api";
import { useAssets, useCatalogIssues, useEnvStatus, useProblems, useRemoveAsset } from "@/lib/api/queries";
import { useAssetFetch } from "@/lib/api/use-asset-fetch";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { useThemeStore } from "@/stores/theme";

const BACKEND_NOTE = {
  tauri: null,
  http: "Connected to the development bridge.",
  fixture: "Showing sample data. Nothing here touches your files.",
} as const;

function SettingsScreen() {
  const env = useEnvStatus();
  const issues = useCatalogIssues();
  const { state: install, install: startInstall } = useEnvInstall();
  const { preference, setPreference } = useThemeStore();
  const assets = useAssets();
  const problems = useProblems();
  const fetch = useAssetFetch();
  const remove = useRemoveAsset();
  const titles = useMemo(
    () => new Map((problems.data ?? []).map((problem) => [problem.id, problem.title])),
    [problems.data],
  );

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

      <Section title="Models" aside="Weights for real-model problems. Nothing downloads until you ask.">
        {assets.error ? (
          <ErrorState
            title="The models could not be listed"
            error={assets.error}
            onRetry={() => void assets.refetch()}
          />
        ) : !assets.data ? (
          <LoadingLines lines={3} />
        ) : (
          <div className="flex flex-col gap-3">
            {remove.error ? (
              <ErrorState title="The model could not be removed" error={remove.error} className="py-0" />
            ) : null}
            <ModelsPanel
              listing={assets.data}
              fetch={fetch}
              titles={titles}
              removing={remove.isPending ? remove.variables : null}
              onRemove={(name) => remove.mutate(name)}
            />
          </div>
        )}
      </Section>

      <Section title="Appearance">
        <ThemePicker value={preference} onChange={setPreference} />
      </Section>

      <Section title="Content" aside="Problems, your solutions, and the Python project live here.">
        {env.data ? (
          <ContentPanel root={env.data.root} issues={issues.data ?? []} />
        ) : (
          <LoadingLines lines={1} />
        )}
      </Section>
    </Page>
  );
}

export const Route = createLazyRoute("/settings")({ component: SettingsScreen });
