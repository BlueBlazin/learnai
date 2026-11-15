import type { CaseInfo, CaseResult } from "@/lib/api";
import { formatNumber } from "@/lib/format";

/** A case as the results list holds it, live or stored. */
export interface ListedCase {
  info: CaseInfo;
  state: "pending" | "running" | "done";
  result: CaseResult | null;
}

export interface StageGroup<C extends ListedCase = ListedCase> {
  /** The stage label as the problem wrote it. */
  label: string;
  /** What the heading says: "Stage <label>", or the label when it already begins with "Stage". */
  heading: string;
  cases: C[];
  passed: number;
  skipped: number;
  total: number;
}

/** "Stage 1: Tokenizer" for "1: Tokenizer"; a label that already says "Stage" is left as written. */
export function stageHeading(label: string): string {
  return /^stage\b/i.test(label.trim()) ? label.trim() : `Stage ${label.trim()}`;
}

/**
 * Groups cases by stage in first-appearance order. Null when no case has a
 * stage: the problem is not staged and its cases are one flat list. In a staged
 * problem a case without a stage goes under "Other cases", after the stages.
 */
export function groupByStage<C extends ListedCase>(cases: C[]): StageGroup<C>[] | null {
  if (!cases.some((c) => c.info.stage)) return null;
  const groups = new Map<string, StageGroup<C>>();
  const unstaged: C[] = [];
  for (const c of cases) {
    const label = c.info.stage;
    if (!label) {
      unstaged.push(c);
      continue;
    }
    let group = groups.get(label);
    if (!group) {
      group = { label, heading: stageHeading(label), cases: [], passed: 0, skipped: 0, total: 0 };
      groups.set(label, group);
    }
    group.cases.push(c);
  }
  const all = [...groups.values()];
  if (unstaged.length > 0) {
    all.push({ label: "", heading: "Other cases", cases: unstaged, passed: 0, skipped: 0, total: 0 });
  }
  for (const group of all) {
    group.total = group.cases.length;
    group.passed = group.cases.filter((c) => c.result?.status === "pass").length;
    group.skipped = group.cases.filter((c) => c.result?.status === "skipped").length;
  }
  return all;
}

/** "3/5 passed", or "3/5 passed, 1 skipped" when any case of the stage was skipped. */
export function stageCount(group: Pick<StageGroup, "passed" | "skipped" | "total">): string {
  const passed = `${formatNumber(group.passed)}/${formatNumber(group.total)} passed`;
  return group.skipped > 0 ? `${passed}, ${formatNumber(group.skipped)} skipped` : passed;
}

/** Whether every case of a stage has passed. */
export function stageComplete(group: Pick<StageGroup, "passed" | "total">): boolean {
  return group.total > 0 && group.passed === group.total;
}
