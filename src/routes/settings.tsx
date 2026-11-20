import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { type ReactNode, useEffect, useMemo } from "react";
import { Page, PageHeader } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { ModelsPanel } from "@/components/models/models-panel";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { FilesPanel } from "@/components/settings/files-panel";
import { GeneralPanel } from "@/components/settings/general-panel";
import { SettingsNav } from "@/components/settings/settings-nav";
import { TargetsPanel } from "@/components/targets/targets-panel";
import { backend } from "@/lib/api";
import {
  useAddTarget,
  useAssets,
  useCatalogIssues,
  useEditorStatus,
  useEnvStatus,
  useProblems,
  useRemoveAsset,
  useRemoveTarget,
  useSetEditorIntelligence,
  useSetReviewsEnabled,
  useSettings,
  useSshHosts,
  useTargets,
} from "@/lib/api/queries";
import { useAssetFetch } from "@/lib/api/use-asset-fetch";
import { useEditorInstall } from "@/lib/api/use-editor-install";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { useTargetCheck } from "@/lib/api/use-target-check";
import { editorSession, useEditorSession } from "@/lib/editor";
import { SETTINGS_SECTION_LABEL } from "@/lib/labels";
import { parseSettingsSection, type SettingsSection } from "@/lib/routes";
import { settingsAttention } from "@/lib/settings";
import { usePreferences } from "@/stores/preferences";
import { useThemeStore } from "@/stores/theme";

const route = getRouteApi("/settings");

const BACKEND_NOTE = {
  tauri: null,
  http: "Connected to the development bridge.",
  fixture: "Showing sample data. Nothing here touches your files.",
} as const;

/** An action that failed, as a line above the list it belongs to. */
function Failure({ title, error }: { title: string; error: unknown }) {
  return error ? <ErrorState title={title} error={error} className="pt-0 pb-4" /> : null;
}

function SettingsScreen() {
  const section = parseSettingsSection(route.useSearch().section);
  // Everything is read here, whatever category is showing: an install or a check keeps
  // its state while the user looks at another category, and the list marks what needs them.
  const env = useEnvStatus();
  const issues = useCatalogIssues();
  const { state: install, install: startInstall } = useEnvInstall();
  const { preference, setPreference } = useThemeStore();
  const assets = useAssets();
  const problems = useProblems();
  const fetch = useAssetFetch();
  const remove = useRemoveAsset();
  const targets = useTargets();
  const hosts = useSshHosts();
  const addTarget = useAddTarget();
  const removeTarget = useRemoveTarget();
  const { checks, check } = useTargetCheck();
  const settings = useSettings();
  const weeklyGoal = usePreferences((state) => state.weeklyGoal);
  const setWeeklyGoal = usePreferences((state) => state.setWeeklyGoal);
  const setReviews = useSetReviewsEnabled();
  const editorStatus = useEditorStatus();
  const setSuggestions = useSetEditorIntelligence();
  const { state: editorInstall, install: installEditor } = useEditorInstall();
  const suggestions = useEditorSession();
  // What this computer is comes from a check. The local check installs nothing, so
  // it runs by itself the first time, once the environment exists to ask.
  const localUnchecked = targets.data?.[0]?.kind === "local" && targets.data[0].lastCheck === null;
  const environmentReady = env.data?.python != null;
  useEffect(() => {
    if (localUnchecked && environmentReady) check("local");
  }, [localUnchecked, environmentReady, check]);
  const titles = useMemo(
    () => new Map((problems.data ?? []).map((problem) => [problem.id, problem.title])),
    [problems.data],
  );
  const attention = settingsAttention({
    env: env.data,
    installFailed: install.status === "failed" || editorInstall.status === "failed",
    assets: assets.data?.assets,
    downloads: fetch.downloads,
    targets: targets.data,
    issues: issues.data,
  });

  const panes: Record<SettingsSection, () => ReactNode> = {
    general: () => (
      <>
        <Failure title="The settings could not be read" error={settings.error ?? editorStatus.error} />
        <Failure title="The setting could not be saved" error={setReviews.error ?? setSuggestions.error} />
        <GeneralPanel
          theme={preference}
          onThemeChange={setPreference}
          weeklyGoal={weeklyGoal}
          onWeeklyGoalChange={setWeeklyGoal}
          reviews={{
            value: settings.data?.reviewsEnabled,
            saving: setReviews.isPending,
            onChange: (enabled) => setReviews.mutate(enabled),
          }}
          suggestions={{
            value: editorStatus.data?.enabled,
            saving: setSuggestions.isPending,
            onChange: (enabled) => setSuggestions.mutate(enabled),
          }}
        />
      </>
    ),
    environment: () =>
      env.error ? (
        <ErrorState
          title="The environment could not be inspected"
          error={env.error}
          onRetry={() => void env.refetch()}
          className="pt-0"
        />
      ) : !env.data ? (
        <LoadingLines lines={6} />
      ) : (
        <>
          <Failure title="The language server could not be inspected" error={editorStatus.error} />
          <EnvironmentPanel
            env={env.data}
            install={install}
            onInstall={(framework) => void startInstall(framework)}
            server={{
              status: editorStatus.data,
              session: suggestions,
              install: editorInstall,
              onInstall: () => void installEditor(),
              onRetry: () => editorSession.restart(),
            }}
          />
        </>
      ),
    models: () =>
      assets.error ? (
        <ErrorState
          title="The models could not be listed"
          error={assets.error}
          onRetry={() => void assets.refetch()}
          className="pt-0"
        />
      ) : !assets.data ? (
        <LoadingLines lines={4} />
      ) : (
        <>
          <Failure title="The model could not be removed" error={remove.error} />
          <ModelsPanel
            listing={assets.data}
            fetch={fetch}
            titles={titles}
            removing={remove.isPending ? remove.variables : null}
            onRemove={(name) => remove.mutate(name)}
          />
        </>
      ),
    targets: () =>
      targets.error ? (
        <ErrorState
          title="The run targets could not be listed"
          error={targets.error}
          onRetry={() => void targets.refetch()}
          className="pt-0"
        />
      ) : !targets.data ? (
        <LoadingLines lines={4} />
      ) : (
        <>
          <Failure title="The machine could not be added" error={addTarget.error} />
          <Failure title="The target could not be removed" error={removeTarget.error} />
          <TargetsPanel
            targets={targets.data}
            hosts={hosts.data ?? []}
            checks={checks}
            adding={addTarget.isPending}
            removing={removeTarget.isPending ? removeTarget.variables : null}
            onCheck={check}
            onAdd={(host) => addTarget.mutate({ host })}
            onRemove={(name) => removeTarget.mutate(name)}
          />
        </>
      ),
    files: () =>
      env.error ? (
        <ErrorState
          title="The content root could not be read"
          error={env.error}
          onRetry={() => void env.refetch()}
          className="pt-0"
        />
      ) : !env.data ? (
        <LoadingLines lines={3} />
      ) : (
        <FilesPanel root={env.data.root} issues={issues.data ?? []} problemCount={problems.data?.length} />
      ),
  };

  return (
    <Page>
      <PageHeader
        title="Settings"
        actions={
          BACKEND_NOTE[backend.kind] ? (
            <p className="text-xs text-muted-foreground">{BACKEND_NOTE[backend.kind]}</p>
          ) : null
        }
      />
      <div className="flex items-start gap-10">
        <SettingsNav current={section} attention={attention} className="sticky top-6 w-44 shrink-0" />
        <section aria-label={SETTINGS_SECTION_LABEL[section]} className="min-w-0 flex-1">
          {panes[section]()}
        </section>
      </div>
    </Page>
  );
}

export const Route = createLazyRoute("/settings")({ component: SettingsScreen });
