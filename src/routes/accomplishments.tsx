import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { SkillDetail } from "@/components/accomplishments/skill-detail";
import { SkillTree } from "@/components/accomplishments/skill-tree";
import { Page, PageHeader } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { useGo } from "@/hooks/use-go";
import { useSkillStates } from "@/hooks/use-skills";
import type { Skill } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";
import { earnedCount } from "@/lib/skills";

const route = getRouteApi("/accomplishments");

/**
 * Accomplishments: the skills as a tree, each earned by solving its problems
 * once the skills it builds on are earned. A tile opens the skill's detail,
 * which is part of the address so that a link can open one.
 */
function AccomplishmentsScreen() {
  const { states, error, retry } = useSkillStates();
  const { skill: openId } = route.useSearch();
  const go = useGo();

  if (error) {
    return (
      <Page wide>
        <ErrorState title="The skills could not be loaded" error={error} onRetry={retry} />
      </Page>
    );
  }
  if (!states) {
    return (
      <Page wide>
        <LoadingLines lines={10} className="max-w-measure" />
      </Page>
    );
  }
  if (states.size === 0) {
    return (
      <Page wide>
        <PageHeader title="Accomplishments" />
        <EmptyState title="There are no skills yet">
          A skill is a file under skills/ in the content root that names the problems that earn it. The tree
          fills in once the content has one.
        </EmptyState>
      </Page>
    );
  }

  const earned = earnedCount(states);
  const open = (skill: Skill) => go(routes.accomplishments(skill.id), { replace: true });
  return (
    <Page wide>
      <PageHeader
        title="Accomplishments"
        description={
          earned === 0
            ? `${pluralize(states.size, "skill")}. Each is earned by solving its problems, and some build on others.`
            : `${earned} of ${states.size} skills earned.`
        }
      />
      <div className="border-t border-rule pt-10">
        <SkillTree states={states} onOpen={open} />
      </div>
      <SkillDetail
        state={openId ? (states.get(openId) ?? null) : null}
        states={states}
        onClose={() => go(routes.accomplishments(), { replace: true })}
        onOpen={open}
      />
    </Page>
  );
}

export const Route = createLazyRoute("/accomplishments")({ component: AccomplishmentsScreen });
