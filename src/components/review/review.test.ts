import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SubmissionList } from "@/components/problem/submission-list";
import { SubmissionView } from "@/components/problem/submission-view";
import { CaseList } from "@/components/results/case-list";
import { ResultsPanel } from "@/components/results/results-panel";
import { ReviewSetting } from "@/components/settings/review-setting";
import type { CaseStatus, Review, Submission, SubmissionDetail } from "@/lib/api";
import { initialRunState } from "@/lib/api/run-state";
import type { ListedCase } from "@/lib/stages";
import { ReviewDueNotice } from "./review-due-notice";
import { ReviewHidden } from "./review-hidden";
import { ReviewList } from "./review-list";
import { ReviewOutcomeNote } from "./review-outcome";
import { ReviewStrip } from "./review-strip";
import { UpcomingReviews } from "./upcoming-reviews";

const NOW = new Date(2026, 10, 15, 10, 0, 0);
const noop = () => {};

const review = (problemId: string, stage: number, dueInDays: number): Review => ({
  problemId,
  stage,
  intervalDays: [3, 7, 21, 60, 180][stage] ?? 180,
  dueAt: new Date(NOW.getTime() + dueInDays * 86_400_000 - 3_600_000).toISOString(),
  lastReviewedAt: null,
  due: dueInDays <= 0,
});

/** The text of rendered markup, with tags dropped and entities decoded. */
const text = (html: string) =>
  html
    .replace(/<!--.*?-->/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/&quot;/g, '"')
    .replace(/&#x27;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/\s+/g, " ")
    .trim();
const render = (element: Parameters<typeof renderToStaticMarkup>[0]) => text(renderToStaticMarkup(element));

const listed = (
  id: string,
  name: string,
  stage: string | undefined,
  status: CaseStatus | "pending",
): ListedCase => ({
  info: { id, name, public: true, ...(stage ? { stage } : {}) },
  state: status === "pending" ? "pending" : "done",
  result:
    status === "pending"
      ? null
      : { id, status, durationMs: 1, message: null, detail: null, stdout: "", metrics: [] },
});

const submission: Submission = {
  id: 7,
  problemId: "stable-softmax",
  framework: "torch",
  mode: "submit",
  verdict: "accepted",
  passed: 5,
  total: 5,
  skipped: 0,
  durationMs: 84,
  target: "local",
  createdAt: "2026-11-15T09:00:00.000Z",
  review: false,
};

describe("review components", () => {
  it("ReviewList names the problem, its track, the review number, and how late it is", () => {
    const out = render(
      createElement(ReviewList, {
        now: NOW,
        onOpen: noop,
        entries: [
          { review: review("a", 4, -3), title: "Numerically Stable Softmax", track: "Numerical Foundations" },
          { review: review("b", 1, 0), title: "Causal Mask" },
        ],
      }),
    );
    expect(out).toContain("Numerically Stable Softmax Numerical Foundations Review 5 of 5 3 days overdue");
    expect(out).toContain("Causal Mask Review 2 of 5 due today");
  });

  it("UpcomingReviews is one quiet line, and nothing when nothing is scheduled", () => {
    expect(
      render(createElement(UpcomingReviews, { now: NOW, reviews: [review("a", 0, 2), review("b", 1, 9)] })),
    ).toBe("2 reviews scheduled; the next is due on Nov 17.");
    expect(
      renderToStaticMarkup(createElement(UpcomingReviews, { now: NOW, reviews: [review("a", 0, 0)] })),
    ).toBe("");
  });

  it("ReviewStrip says what the review is and offers every way out", () => {
    const actions = { onForgot: noop, onSnooze: noop, onStartOver: noop, onRetire: noop, onLeave: noop };
    const html = renderToStaticMarkup(createElement(ReviewStrip, { review: review("a", 1, 0), ...actions }));
    const out = text(html);
    expect(out).toContain("Review 2 of 5. Solve it again from the blank starter.");
    for (const label of [
      "I forgot",
      "Snooze a day",
      "Start over",
      "Stop reviewing this problem",
      "Leave review",
    ]) {
      expect(out).toContain(label);
    }
    expect(html).not.toContain(' disabled=""');

    // During a run nothing can be pressed; a failed action is said in the strip.
    const busy = renderToStaticMarkup(
      createElement(ReviewStrip, { review: review("a", 1, 0), busy: true, ...actions }),
    );
    expect(busy.match(/ disabled=""/g)?.length).toBe(5);
    expect(
      render(
        createElement(ReviewStrip, { review: review("a", 1, 0), error: "database is locked", ...actions }),
      ),
    ).toContain("database is locked");
  });

  it("ReviewDueNotice offers to start the review", () => {
    const out = render(
      createElement(ReviewDueNotice, { review: review("a", 2, -1), now: NOW, onStart: noop }),
    );
    expect(out).toContain("This problem is up for review (review 3 of 5, 1 day overdue).");
    expect(out).toContain("Start review");
  });

  it("ReviewHidden says what is hidden and how to get it back", () => {
    const out = render(createElement(ReviewHidden, { what: "The hints are" }));
    expect(out).toContain("Hidden during a review");
    expect(out).toContain(
      'The hints are out of sight until the review is over. If you are stuck, choose "I forgot"',
    );
  });

  it("ReviewOutcomeNote says when the next review is due, or that the problem is mastered", () => {
    expect(
      render(
        createElement(ReviewOutcomeNote, {
          now: NOW,
          outcome: { kind: "advanced", label: "Review 1 of 5", next: review("a", 1, 7) },
        }),
      ),
    ).toBe("Review 1 of 5 done. The next review is due on Nov 22, in 7 days.");
    expect(
      render(createElement(ReviewOutcomeNote, { outcome: { kind: "mastered", label: "Review 5 of 5" } })),
    ).toContain("this problem is mastered");
  });

  it("ReviewSetting is a labeled switch with one line of help", () => {
    const on = renderToStaticMarkup(createElement(ReviewSetting, { enabled: true, onChange: noop }));
    expect(on).toContain('role="switch"');
    expect(on).toContain('aria-checked="true"');
    expect(on).toMatch(/<label[^>]*>Reviews<\/label>/);
    expect(text(on)).toContain("after 3, 7, 21, 60, and 180 days");
    const off = renderToStaticMarkup(createElement(ReviewSetting, { enabled: false, onChange: noop }));
    expect(off).toContain('aria-checked="false"');
    // Off, the line says what became of the schedules instead.
    expect(text(off)).toContain("schedules are kept");
    expect(text(off)).not.toContain("after 3, 7, 21, 60, and 180 days");
  });
});

describe("review marks in history and results", () => {
  it("SubmissionList marks review attempts and nothing else", () => {
    const out = render(
      createElement(SubmissionList, {
        onOpen: noop,
        submissions: [{ ...submission, id: 8, review: true }, submission],
      }),
    );
    expect(out.match(/Review/g)?.length).toBe(1);
  });

  it("SubmissionView hides the code and case details during a review", () => {
    const detail: SubmissionDetail = {
      ...submission,
      review: true,
      code: "def softmax(x):\n    return SECRET\n",
      cases: [
        {
          id: "c1",
          name: "small vector",
          public: true,
          status: "pass",
          durationMs: 1,
          message: null,
          detail: null,
          stdout: "",
          metrics: [],
        },
      ],
      metrics: [],
    };
    const hidden = render(
      createElement(SubmissionView, { submission: detail, onBack: noop, codeHidden: true }),
    );
    expect(hidden).toContain("Accepted");
    expect(hidden).toContain("Review, from the blank starter");
    expect(hidden).toContain("Hidden during a review");
    expect(hidden).not.toContain("SECRET");
    expect(hidden).not.toContain("small vector");
    expect(hidden).not.toContain("Code as submitted");

    const shown = render(createElement(SubmissionView, { submission: detail, onBack: noop }));
    expect(shown).toContain("small vector");
    expect(shown).toContain("Code as submitted");
    expect(shown).not.toContain("Hidden during a review");
  });

  it("ResultsPanel shows the note under a finished verdict only", () => {
    const note = createElement("p", null, "NOTE");
    const done = {
      ...initialRunState,
      phase: "done" as const,
      mode: "submit" as const,
      verdict: "accepted" as const,
    };
    expect(render(createElement(ResultsPanel, { run: done, onCancel: noop, note }))).toContain("NOTE");
    const running = { ...initialRunState, phase: "running" as const, mode: "submit" as const };
    expect(render(createElement(ResultsPanel, { run: running, onCancel: noop, note }))).not.toContain("NOTE");
    expect(
      render(createElement(ResultsPanel, { run: initialRunState, onCancel: noop, idleHint: "Review hint." })),
    ).toContain("Review hint.");
  });
});

describe("CaseList with stages", () => {
  const staged = [
    listed("t1", "round-trips ASCII", "1: Tokenizer", "pass"),
    listed("t2", "round-trips multi-byte text", "1: Tokenizer", "pass"),
    listed("d1", "greedy decode", "Stage 2: Decode loop", "fail"),
    listed("d2", "Hidden case 2", "Stage 2: Decode loop", "skipped"),
    listed("s1", "Hidden case 3", "3: Serving", "pending"),
  ];

  it("sets the cases under stage headings with a count each", () => {
    const html = renderToStaticMarkup(createElement(CaseList, { cases: staged, ended: false }));
    const out = text(html);
    expect(html.match(/<section/g)?.length).toBe(3);
    expect(html.match(/<h3/g)?.length).toBe(3);
    expect(out).toContain("Stage 1: Tokenizer 2/2 passed");
    // A label that already begins with "Stage" is shown as written.
    expect(out).toContain("Stage 2: Decode loop 0/2 passed, 1 skipped");
    expect(out).not.toContain("Stage Stage");
    expect(out).toContain("Stage 3: Serving 0/1 passed");
    // The order is the order of first appearance.
    expect(out.indexOf("Stage 1: Tokenizer")).toBeLessThan(out.indexOf("Stage 2: Decode loop"));
    expect(out.indexOf("Stage 2: Decode loop")).toBeLessThan(out.indexOf("Stage 3: Serving"));
    expect(out.indexOf("round-trips ASCII")).toBeLessThan(out.indexOf("Stage 2: Decode loop"));
  });

  it("renders an unstaged problem as one flat list, as before", () => {
    const flat = staged.map((c) => ({ ...c, info: { id: c.info.id, name: c.info.name, public: true } }));
    const html = renderToStaticMarkup(createElement(CaseList, { cases: flat, ended: true }));
    expect(html).not.toContain("<section");
    expect(html).not.toContain("<h3");
    expect(html.match(/<ol/g)?.length).toBe(1);
    expect(text(html)).not.toContain("passed");
  });
});
