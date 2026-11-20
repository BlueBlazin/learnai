import { createLazyRoute } from "@tanstack/react-router";
import { useEffect, useMemo } from "react";
import { Page, PageHeader, Section } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { ModelsPanel } from "@/components/models/models-panel";
import { ContentPanel } from "@/components/settings/content-panel";
import { EnvironmentPanel } from "@/components/settings/environment-panel";
import { GoalSetting } from "@/components/settings/goal-setting";
import { ReviewSetting } from "@/components/settings/review-setting";
import { SuggestionsPanel } from "@/components/settings/suggestions-panel";
import { ThemePicker } from "@/components/settings/theme-picker";
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
import { usePreferences } from "@/stores/preferences";
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

      <Section
        title="Run targets"
        aside="Where the judge runs: this computer, or a machine of yours reached over SSH."
      >
        {targets.error ? (
          <ErrorState
            title="The run targets could not be listed"
            error={targets.error}
            onRetry={() => void targets.refetch()}
          />
        ) : !targets.data ? (
          <LoadingLines lines={3} />
        ) : (
          <div className="flex flex-col gap-3">
            {addTarget.error ? (
              <ErrorState title="The machine could not be added" error={addTarget.error} className="py-0" />
            ) : null}
            {removeTarget.error ? (
              <ErrorState
                title="The target could not be removed"
                error={removeTarget.error}
                className="py-0"
              />
            ) : null}
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
          </div>
        )}
      </Section>

      <Section title="Review" aside="Solving a problem once is not the same as keeping it.">
        {settings.error ? (
          <ErrorState
            title="The settings could not be read"
            error={settings.error}
            onRetry={() => void settings.refetch()}
          />
        ) : !settings.data ? (
          <LoadingLines lines={2} />
        ) : (
          <div className="flex flex-col gap-3">
            {setReviews.error ? (
              <ErrorState title="The setting could not be saved" error={setReviews.error} className="py-0" />
            ) : null}
            <ReviewSetting
              enabled={settings.data.reviewsEnabled}
              disabled={setReviews.isPending}
              onChange={(enabled) => setReviews.mutate(enabled)}
            />
          </div>
        )}
      </Section>

      <Section title="Goal" aside="How many problems you aim to solve each week.">
        <GoalSetting value={weeklyGoal} onChange={setWeeklyGoal} />
      </Section>

      <Section title="Editor" aside="Suggestions and documentation while you write a solution.">
        {editorStatus.error ? (
          <ErrorState
            title="The editor settings could not be read"
            error={editorStatus.error}
            onRetry={() => void editorStatus.refetch()}
          />
        ) : !editorStatus.data ? (
          <LoadingLines lines={3} />
        ) : (
          <div className="flex flex-col gap-3">
            {setSuggestions.error ? (
              <ErrorState
                title="The setting could not be saved"
                error={setSuggestions.error}
                className="py-0"
              />
            ) : null}
            <SuggestionsPanel
              status={editorStatus.data}
              session={suggestions}
              saving={setSuggestions.isPending}
              onToggle={(enabled) => setSuggestions.mutate(enabled)}
              install={editorInstall}
              onInstall={() => void installEditor()}
              onRetry={() => editorSession.restart()}
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
