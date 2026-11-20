# Problem production process

How a problem gets from an idea into the catalog: research, specification, kit work,
authoring, verification, and commit. This is the working procedure for the next batch.
It records what was done for the first 213 problems (2026-11-14 to 2026-11-16), recovered
from the session transcripts, the commit history, and the briefs that were sent.

What a problem *is* (files, test API, judges, kit, house style, tolerances, mutants) is in
[`PROBLEM_AUTHORING.md`](PROBLEM_AUTHORING.md). The catalog and its specs are in
[`CURRICULUM.md`](CURRICULUM.md). This document does not repeat either.

Two labels are used throughout:

- **[Recommended]** marks a step that was *not* past practice. It is proposed here because
  of a defect or a gap listed in sections 7 and 8. Everything unmarked was done.
- **[Tooling]** marks a check that a command enforces. **[Judgment]** marks a check that
  relied on the author or the manager reading something.

All commands run from the repository root. `learnai` below abbreviates

```sh
uv run --project python --extra jax --extra mlx --extra models --extra editor learnai
```

and `ruff` abbreviates the same prefix with `ruff` in place of `learnai`. Always pass the
extras.

## 1. Roles and standing rules

| role | who | does |
|---|---|---|
| owner | the person | sets scope and standards, answers the decisions the manager brings, reviews at checkpoints |
| manager | the main session | plans, writes briefs, runs one subagent at a time, re-checks each batch, commits |
| researcher | subagent | surveys primary sources and writes or extends the catalog specs |
| kit engineer | subagent | extends `python/` (kit, judge) and its tests before the batches that need it |
| author | subagent | writes one batch of problems and verifies them |
| reviewer | subagent | **[Recommended]** reads a finished batch cold; see section 4, step 6 |

Rules the owner set, in the owner's words where they set a standard:

- **Orchestration.** "Spawn Fable 5.2 subagents at Medium thinking effort ... Instead of
  reusing spawned subagents for new tasks, spawn new subagents to keep context clean. You
  act as the manager and orchestrator, stepping in if necessary." Every problem subagent so
  far was a general-purpose Fable 5.2 agent at medium effort. An interrupted subagent was
  resumed, not replaced.
- **One at a time.** "Having so many subagents is eating through my quota. Can you pause
  all but 1 and go sequentially instead??" Milestones 1 and 2 ran up to nine authors in
  parallel; everything after ran one subagent at a time. Sequential is the rule.
- **Cheaper models.** "For well defined tasks, you can actually even use Sonnet 5.5
  subagents and then verify everything yourself or using a verified Fable 5.2 subagent."
  (2026-11-20; no problem has been authored this way yet.)
- **Currency.** "Always use web search and official docs to find out the latest version of
  libraries so you do not use outdated versions." For models: "you won't get very far with
  those outdated models. I am ok with models up to 4B params (though at 4B I prefer 8-bit
  quantized to manage weights). But preferably safetensors only."
- **Design first.** "Don't just go off one-shotting it. First propose a well thought out
  design to me."
- **Git.** Local commits per step; never push.
- **Hardware.** Apple M1 Max, 64 GB, no NVIDIA GPU: "just add unit tests for now".

## 2. The pipeline at a glance

| # | step | who | output |
|---|---|---|---|
| 1 | Research | researcher, then manager | sourced inventory; facts marked verified or not |
| 2 | Specification | researcher writes, manager reads, owner decides open points | rows and spec paragraphs in `CURRICULUM.md`, committed |
| 3 | Kit and judge work | kit engineer, then manager | kit objects, tests, guide sections, committed |
| 4 | Authoring | author | one batch of complete problems, uncommitted, with a report |
| 5 | Batch acceptance | manager | batch re-validated and committed |
| 6 | Independent review | reviewer | **[Recommended]** findings fixed before the commit |
| 7 | Closing pass | manager | whole catalog validated, suites green, figures seen in the app |

## 3. Research and specification

### Step 1. Research

- **Who.** A researcher subagent; the manager compares the result with the catalog.
- **In.** The owner's goal in their own words, the tracks and counts wanted, the existing
  catalog, and what the judge and kit can express (`ARCHITECTURE.md` section 3,
  `PROBLEM_AUTHORING.md`).
- **How.** By web search and reading, not from memory ("your training data is stale" was
  in every research brief). Sources in order of trust: a model's `config.json` and model
  card, the paper or technical report, official docs, the reference implementation
  (`transformers`, `mlx-lm`, vLLM, llama.cpp). News and social posts are leads only.
  Where an engine and a paper disagree, the spec says which one the problem follows.
- **Out.** Every claim labeled as verified against a primary source (with the URL) or as
  secondary or inferred. The catalog's record of this is the
  [Sources](CURRICULUM.md#sources) section of `CURRICULUM.md`, which dates the check and
  lists what was *not* re-verified. Topics that are outdated, removed upstream, or not
  judgeable deterministically are dropped and listed
  ([section 7](CURRICULUM.md#7-coverage-notes-and-deliberate-omissions)).
- **Limit of this step.** Research establishes what exists and where it is specified. It
  does not settle the mechanics: the 2026-11-20 architecture survey states that it read
  configs and cards but no modeling code. The author re-verifies each named source when
  writing the problem (step 4).

**Library versions and house models** were confirmed by doing, not by reading alone:
versions from live registry queries, then installed (`VERSIONS.md`); candidate models
found by search, then downloaded, loaded in `transformers` and `mlx-lm`, and matched with
a from-scratch forward pass (largest logit error 8.4e-5) before the choice was final. The
verified facts are in `PROBLEM_AUTHORING.md` section 15 and supersede the curriculum
where they differ. Assets are pinned by a full 40-character revision.

### Step 2. Specification

- **Who.** The researcher writes the specs; the manager reads the whole document and
  commits it; the owner decides what the manager brings.
- **Format.** One table row and one spec paragraph of two to four sentences per problem,
  as defined in [`CURRICULUM.md` section 1](CURRICULUM.md#1-how-to-read-this-document):
  what the learner implements (named function or interface), the learning goal, how it is
  judged, the mutants, the named source, and `[kit: X]` for kit that does not exist. The
  bar from the brief: "specific enough that two different authors would write essentially
  the same problem."
- **Also delivered.** Kit additions with proposed interfaces
  ([section 3](CURRICULUM.md#3-kit-additions-needed)), the partition into author batches
  ([section 5](CURRICULUM.md#5-author-batches-for-m2)), and the open decisions
  ([section 8](CURRICULUM.md#8-decisions-and-risks-for-the-owner)).
- **Script-checked [Tooling, ad hoc].** Unique ids, every prerequisite exists, every
  problem has a spec, batches partition the set. This was a throwaway script by the
  researcher; nothing in the repository repeats it.
- **Owner's part, as practiced.** The owner chose scope, frameworks, the model size limit,
  and a review checkpoint after the first 12 problems ("Milestone 2 copies this statement
  style, difficulty, and hint depth about 70 more times, so changes are cheapest now";
  the answer was "Yes, continue"). The owner did not review the 213 specs one by one. The
  curriculum's five open decisions were recorded in section 8 and settled by the manager,
  the model choice by testing.
- **[Recommended]** Because the owner asked to see a design before a large build, show
  the owner the list of new problems (id, one line, source, real model if any) and the
  open decisions before authoring starts, and record the date each spec's source was
  checked.
- **Where the new specs go.** Add the track table and specs to `CURRICULUM.md`, a
  `track.toml` per new module (with `core`), and the module in an `areas/<id>.toml`
  (`ARCHITECTURE.md` section 3.3).

## 4. Building a batch

### Step 3. Kit and judge work first

- **Who.** One kit engineer subagent per milestone, before any author who needs it.
- **In.** The "needed" kit list from the specs, every spec paragraph that names a kit
  object ("let the problem specs drive each interface"), and **the judge, kit, and guide
  problems that the previous authors reported**. Each milestone's kit brief opened with
  that list of fixes.
- **Out.** Kit objects under `python/src/learnai/kit/`, unit tests under `python/tests/`,
  and a section in `PROBLEM_AUTHORING.md` (rules:
  [Adding a kit object](PROBLEM_AUTHORING.md#adding-a-kit-object)). Library behavior the
  kit imitates is checked against official docs *and* by running it, and what could not
  be verified is written down in the guide.
- **Definition of done.**

  ```sh
  uv run --project python --extra jax --extra mlx --extra models --extra editor pytest python/tests
  learnai validate --all          # three times
  ruff check python
  ```

- **Manager.** Re-runs `pytest` and `learnai validate --all`, mirrors contract changes
  into `ARCHITECTURE.md`, commits.

### Step 4. Authoring

- **Unit of work.** One author subagent per batch; a batch is problems that share a track
  or kit knowledge. Sizes used: 3 or 4 (first 12), 4 or 5 (milestones 2 to 4), 4 to 6
  (milestone 5), 2 per batch for capstones. A milestone 5 batch took about 30 to 40
  minutes.
- **Brief.** A shared brief plus a short per-batch message (section 5). From milestone 5
  the shared brief was a file the manager wrote once and amended when a batch exposed a
  problem; each author was told to read it first.
- **What the author does, in order.**
  1. Reads the guide sections named in the brief, the exemplar
     `problems/foundations/stable-softmax/` in full, the two or three nearest existing
     problems, each prerequisite, and its rows and spec paragraphs in `CURRICULUM.md`.
  2. Verifies each spec against its named source by web search or by reading the code or
     paper, and against the installed library by running it. Where the spec is wrong,
     ambiguous, or names a mutant that cannot fail, the author fixes it in the problem,
     keeps the id and the learning goal, and reports the change.
  3. Scaffolds with `learnai new <track>/<id>` and writes every file
     ([directory format](PROBLEM_AUTHORING.md#2-directory-format),
     [house style](PROBLEM_AUTHORING.md#8-statement-house-style),
     [mutants](PROBLEM_AUTHORING.md#11-mutants)). Worked examples use numbers obtained
     by running the reference, "never invented". Reference outputs for real-model
     problems come from a committed `fixtures/make.py` that uses the real libraries, not
     from the reference solution.
  4. Runs the checks of section 4.1 and the
     [guide's checklist](PROBLEM_AUTHORING.md#16-checklist-before-you-open-a-pull-request).
  5. Reports in under 250 words: per problem the public and hidden case counts, mutant
     count, and submit time per framework; sources verified; **sources quoted from
     memory**; spec corrections; judge, kit, or guide problems; anything flaky, slow, or
     not done. Does not commit.
- **Boundaries.** Only its own problem directories. Never `python/`, `docs/`, other
  problems, or the model directory. A kit gap is worked around with a local helper in
  `tests.py` and reported. No subagents of its own.

### 4.1 The checks, in order

| # | check | command or method | enforced by |
|---|---|---|---|
| 1 | Metadata, files, `forbidden` names, no `TODO(author)` | `learnai validate <id> -v` | [Tooling] |
| 2 | Every reference accepted on a full submit (public and hidden cases), in each listed framework | same | [Tooling] |
| 3 | Frameworks agree: the JAX and MLX references are judged against the torch oracle within the case tolerances | same | [Tooling] |
| 4 | Every starter runs and is rejected | same | [Tooling] |
| 5 | Every mutant rejected | same | [Tooling] |
| 6 | Not flaky: the validation repeated, two times for numeric problems, three for simulated-clock, service, speed, GPU cost, and memory cases | repeat `learnai validate <id>` | [Judgment] (author reports the count) |
| 7 | Time: no "cases took more than 3 s" warning; statistical cases under about 3 s in every framework | `learnai validate <id> -v`, `learnai submit <id> -f <fw> --reference` | warning only; [Judgment] |
| 8 | Lint | `ruff check problems/<track>/<id>` | [Tooling] |
| 9 | Fairness: at least one alternative correct solution per problem is accepted | `learnai submit <id> -f <fw> --solution <path>` | [Judgment] |
| 10 | Each mutant fails for the reason on its `# BUG:` line, and hidden failures leak nothing | `learnai submit <id> --mutant <name>` with and without `--reveal` | [Judgment] |
| 11 | The starter's failure is one a beginner can act on | `learnai run <id> -f <fw> --starter` | [Judgment] |
| 12 | Statement complete and unambiguous; editorial teaches and cites; figures readable in light and dark | reading; the app | [Judgment] |
| 13 | Sources: every formula and system behavior checked against the named source | web search, reading code | [Judgment] |

What the terms mean here. A **mutant** is a wrong solution in the oracle framework with
one realistic bug; it is **killed** (rejected) when a full submit ends `wrong_answer`,
`runtime_error`, or `time_limit`. Mutants run only in the oracle framework. A mutant that
survives means a test is missing: add the case, never delete the mutant.
[`validate` in full](PROBLEM_AUTHORING.md#what-learnai-validate-enforces).

**Determinism** is by construction (float32, randomness as an input, simulated clocks:
[guide section 3](PROBLEM_AUTHORING.md#3-the-two-rules-that-keep-judging-exact)) and is
observed only through check 6. `validate` runs speed and service problems serially after
the parallel batch, so `judges` must list every judge type used.

**When a check fails.** The author fixes the problem, not the check: a surviving mutant
gets a new case; a mutant that cannot fail is replaced by one that can, and the report
says so; a tolerance is not loosened to turn a case green; a slow JAX case gets fewer
distinct shapes or less work, not a higher limit; a spec that contradicts its source is
corrected in the problem. A judge or kit bug is reported, not patched, and goes to the
next kit engineer (step 3).

### 4.2 Problems that need weights or hardware

| kind | how it was verified on the owner's machine | never verified |
|---|---|---|
| Real model (`[[assets]]`) | Weights on disk under `.learnai/models/`. `learnai validate <id> --require-assets` so nothing is skipped, twice per framework. `LEARNAI_MODELS_DIR=/nonexistent learnai submit <id> --reference`: synthetic cases run, the verdict is "assets missing", the starter is still rejected. Fixtures are small, committed, and regenerated by `fixtures/make.py` (needs `--extra models`). At least one mutant is catchable only by a real-weight case. | nothing |
| MLX-only (`requires = "mlx-gpu"`) | Runs on this machine's GPU. Every API checked against the MLX docs for the installed version and by running it. Validated three or four times because cost and memory cases must be stable; once with `LEARNAI_HARDWARE=cpu`. Cost cases are exact counts, never wall-clock thresholds. | nothing |
| CUDA (`requires = "cuda"`) | CPU logic cases decide validation here, using the kit stand-ins of [guide section 18](PROBLEM_AUTHORING.md#18-cuda-problems). **Every mutant must be rejected by a CPU case.** A reference ends "hardware missing" with all CPU cases passing. `LEARNAI_HARDWARE=cpu,cuda` must not crash the harness. GPU cases are few, simple, written from the PyTorch docs, and the statement says which facts were checked only on the CPU. | every `requires="cuda"` case; the Triton shim against real Triton |

`learnai validate` reports what it could not see on every run ("14 passed with cases that
could NOT be validated on this machine").

### Step 5. Batch acceptance and commit (manager)

Done for a batch meant: the author's report claims every check of 4.1, and the manager has
done the following.

1. Read the report: spec corrections, anything "from memory", anything "not done".
2. Re-run the gate on exactly the batch **[Tooling]**:

   ```sh
   learnai validate <id> <id> ...
   git status --short
   ```

   Every changed path must be inside the batch's directories.
3. Spot-check **[Judgment]**: read part of a statement; for a capstone, run a reference
   submit and confirm the per-stage counts; check a doubtful citation by web search (one
   editorial's author list was removed this way).
4. Commit the batch alone: its problem directories and a new `track.toml` if any. The
   message names the batch and lists the ids, for example
   `Problems: frontier track, first batch for milestone 5 (203)` with the count of
   problems after the commit. Nothing else rides along.
5. Carry reported judge, kit, and guide problems to the next kit engineer, and add any
   new process rule to the shared brief before the next author starts.

In milestone 2 (parallel authors) the manager committed on the author's report and ran
`learnai validate --all` only at the end of the milestone. From milestone 3 the re-run in
item 2 preceded every commit. Item 2 before every commit is the rule.

### Step 6. Independent review **[Recommended]**

No problem was ever read by a second subagent. The manager's spot checks were the only
review, and the statement, hints, editorial, and failure messages of most problems were
read only by their author. For the next batch, before the commit of step 5, give each
batch to a fresh Fable 5.2 reviewer with the brief in section 5.3, have the manager (or
the author, resumed) fix what it finds, and re-run step 5 item 2.

### Step 7. Closing pass (manager)

After the last batch of a milestone:

```sh
learnai validate --all             # every problem; read every warning
uv run --project python --extra jax --extra mlx --extra models --extra editor pytest python/tests
ruff check python problems
```

Then: look at the new figures in the app in light and dark (`pnpm dev:real`), grep new
SVGs for hard-coded colors, update `CURRICULUM.md` totals and each module's `core` list,
and tell the owner what was *not* verified. **[Recommended]** Also write the authors'
spec corrections back into `CURRICULUM.md` (see section 8).

## 5. Brief templates

Distilled from the briefs that were sent. Fill the braces.

### 5.1 Author, shared part

```text
You are a problem author for Learn AI, a local desktop app for practicing LLM inference
and serving (a LeetCode-style judge). The repository is /Users/.../learnai. The catalog
has {N} validated problems. You author one batch. The learner is a strong engineer
practicing for mastery, so problems must be precise, realistic, and worth solving twice.
Write American English everywhere. Today is {date}; your training data is stale.

The ids, titles, orders, difficulties, judges, frameworks, and prerequisites of your
problems are fixed by docs/CURRICULUM.md (table row plus spec paragraph per id).

## Read first
- docs/PROBLEM_AUTHORING.md: sections 1-4 and 8-12, the kit entries you need in section
  7, the checklist in section 16, {section 20 for case stages; 13-15 and 18 only for
  MLX-only, real-model, or CUDA problems}.
- The exemplar problems/foundations/stable-softmax/ in full, then the two or three
  existing problems nearest to each of yours (same track, and each prerequisite), so your
  statements, tests, and mutants match the house style and do not repeat them.
- docs/CURRICULUM.md section 1, and for each of your ids its table row and spec paragraph.
- For any technique from a paper or a real system, check the paper, the official docs, or
  the code with web search rather than recalling; state the source in the editorial. Do
  not invent behavior of real systems. Cite a paper by title and arXiv id or venue only
  when you have checked it exists; do not name authors from memory.

## Facts verified by earlier batches
{versions and machine: see docs/VERSIONS.md; framework quirks from PROBLEM_AUTHORING
sections 10, 13, 18; for example: MLX boolean-mask reads fail; JAX costs 0.15-0.2 s per
new input shape}
- Speed cases are relative to the reference, never absolute; aim for the wrong approach
  being 5x or more slower against a limit of about 3x. Measure, and size the case to what
  you measure. Cost cases must be deterministic.
- Random draws come from supplied uniforms, not from the framework's generator.
- Where the curriculum's spec turns out wrong or unobservable, fix the problem (keep the
  id and the learning goal) and report the change.

## Standards
- Statement: precise, in the house style. Every behavior the tests check is stated. The
  worked example uses numbers you computed by running the reference, never invented.
- Starter: runs, is not accepted, and gives nothing away. Reference: the clearest correct
  solution per framework.
- Hints: graded, from a nudge to nearly the full idea. Editorial: the idea, the
  derivation or invariant, the common mistakes (your mutants), the cost, and where it
  appears in real systems, with sources.
- Mutants: the bugs people actually write, at least the ones the curriculum names; every
  mutant rejected, each by a case that names the mistake.
- Tests tight enough that every mutant fails and fair enough that any correct solution
  passes in every framework at float32: try at least one alternative correct solution
  per problem.
- Public cases let the learner debug; hidden cases stop hard-coding. Failure messages say
  what went wrong in terms the learner can act on and leak no expected values.
- A figure (SVG, currentColor) only where it carries the idea.

## Commands (from the repository root; always pass the extras)
- uv run --project python --extra jax --extra mlx --extra models --extra editor learnai validate <id>
  Validate every problem two or three times. Done means the reference is accepted, the
  starter rejected, and every mutant rejected, in every framework, with no warning.
- ... ruff check problems/<track>/<id>, one path per problem, never a whole track.
- Before you report: git status --short; every changed path must be inside your own
  problem directories. Restore anything else with git checkout -- <path>.

## Boundaries
- Create and edit only your own problem directories. Do not edit the judge, the kit,
  docs/, other problems, or the app. If the kit lacks something, write a small local
  helper in your tests.py and report it.
- Keep scratch files in your own folder: {scratchpad}/{batch-name}/.
- Do not commit, push, install global tools, or spawn subagents. Do not touch
  .learnai/models/. Finish or stop any background work before you report.

## Report (under 250 words)
Per problem: one line with what it teaches, public/hidden case counts, mutant count, and
reference submit time per framework. Then: sources you verified and sources you quoted
from memory, spec corrections, kit gaps, anything flaky or slow, anything not done.
```

### 5.2 Author, per-batch part

```text
Your batch ({n} problems; ids, orders, and specs are in docs/CURRICULUM.md):
- Track {track} (problems/{track}/): {id}, {id}, ...

Notes for this batch:
- {id}: {the one thing most likely to go wrong: a library behavior to measure first, an
  edge case to define exactly, a mutant that must be observable, a source whose rule must
  match exactly, an existing problem whose conventions this one must keep}.
- {order to write them in, if later ones build on earlier ones}.

Done means all {n} pass learnai validate repeatedly and ruff is clean. Do not commit.
```

Additions by kind, as they were sent:

- **Real model.** Name the asset and say it is on disk and must not be deleted or
  re-downloaded. Public cases on small synthetic inputs marked `assets=[]` so a learner
  can start without the download. Fixtures from `fixtures/make.py` using the real
  libraries. Forbid `transformers`, `tokenizers`, `safetensors`, and the kit oracle in the
  solution. List the model facts verified earlier and add "verify against the files rather
  than trusting this list". Ask for measured submit times per framework.
- **MLX-only.** "Check every API you teach against the official docs and by running it."
  Validate three times, and once with `LEARNAI_HARDWARE=cpu`. Results must also be
  checked numerically; "a cost-only problem is not acceptable".
- **CUDA.** "Every mutant must be rejected by the CPU cases on this machine." GPU
  semantics that cannot be run "must be stated from the docs and marked as such in the
  editorial". Ask which GPU cases exist per problem.
- **Capstone.** Two per batch, `time_limit_s = 30`, every case staged
  ([guide section 20](PROBLEM_AUTHORING.md#20-contract-additions-of-milestone-5-to-mirror-in-architecturemd)).
  A learner must pass stage N without implementing later stages; at least one mutant per
  stage; the statement keeps the rules of the prerequisite problems. Name an existing
  capstone to read in full.

### 5.3 Reviewer **[Recommended]**

Not past practice. The checks are the [Judgment] rows of section 4.1 and the defects of
section 7.

```text
You are reviewing a batch of new Learn AI practice problems before they are committed.
Repository: /Users/.../learnai. You did not write them; judge them as a learner and as a
skeptical engineer. Write American English. Change no file. Do not spawn subagents.

Batch: {ids with paths}. Their specs: docs/CURRICULUM.md (grep each id). The standard:
docs/PROBLEM_AUTHORING.md sections 3, 8, 9, 11, and the checklist in section 16; the
exemplar problems/foundations/stable-softmax/.

For each problem:
1. Read statement.md first, alone. List every place where two careful readers could
   implement different things and both be right (ties, empty input, dtype, rounding,
   ordering, units). Then read tests.py: is every tested behavior stated?
2. Sources. For each formula, constant, and claim about a real model or system in the
   statement and editorial, open the named primary source and compare. Report mismatches,
   claims with no source, and citations you could not find. Do not trust the author.
3. Solve it yourself from the statement in one framework without reading reference/, and
   submit it: learnai submit <id> -f <fw> --solution <path>. A rejected honest solution
   is a finding. Keep your file under {scratchpad}/{review-name}/.
4. Mutants. Read each mutant against its "# BUG:" line; run learnai submit <id> --mutant
   <name> --reveal for each. Report mutants that fail for a different reason than
   stated, that only crash, or that fail every case. Then name one realistic wrong
   solution that no mutant covers; write it and submit it. If it is accepted, that is the
   most important finding.
5. Run learnai submit <id> --mutant <one> without --reveal: do hidden cases reveal
   expected values, sizes, or what they test? Run learnai run <id> -f <fw> --starter: is
   the first failure actionable?
6. Hints and editorial: do the hints progress without giving the answer; does the
   editorial explain why, cover each mutant, and state cost?
7. Run learnai validate <id> -v three times; report any warning or difference between
   runs. Run ruff check problems/<track>/<id>.

Report per problem: findings ordered by severity (wrong fact, unfair test, hole in the
tests, ambiguity, leak, style), each with the file and the evidence, and what you checked
and found correct. Say plainly what you did not check.
```

## 6. Acceptance checklists

### One problem

- [ ] `learnai validate <id> -v` passes two or three times in a row, with no warning (or
      the warning is a skipped CUDA case, or a timing warning the report explains).
- [ ] `ruff check problems/<track>/<id>` is clean.
- [ ] Complete: `problem.toml`, statement, starter and reference per framework, tests with
      public and hidden cases, three or more graded hints, editorial, four or more mutants
      (the catalog has 5 to 22, median 9), and a `forbidden` list with a `uses_builtin`
      mutant wherever the statement rules out a built-in.
- [ ] The first public case is the statement's worked example; its numbers came from
      running the reference.
- [ ] Every mutant the spec names exists or the report says why it could not fail and what
      replaced it; each mutant is rejected by a case written for it.
- [ ] At least one alternative correct solution was accepted.
- [ ] Every named source was checked at authoring time; anything from memory is listed in
      the report and is not stated as fact about a real system.
- [ ] Every item of the
      [guide's checklist](PROBLEM_AUTHORING.md#16-checklist-before-you-open-a-pull-request)
      that applies (hardware, assets, memory limits, statistical time, hidden-case leaks).
- [ ] **[Recommended]** A reviewer who did not write it solved it from the statement and
      found no accepted wrong solution.

### One batch

- [ ] The kit it needs was committed before the author started.
- [ ] The author's report is in hand and its spec corrections and "not done" items are
      read.
- [ ] The manager re-ran `learnai validate <ids>` after the author finished.
- [ ] `git status --short` shows changes only inside the batch's directories.
- [ ] New figures were seen in the app in light and dark, or are listed as not yet seen.
- [ ] The commit contains only the batch; the message lists the ids.
- [ ] Reported judge, kit, and guide problems are queued for the next kit engineer; new
      process rules are in the shared brief.
- [ ] **[Recommended]** Spec corrections are written back to `CURRICULUM.md`.

### Closing a set of batches

- [ ] `learnai validate --all` passes; every warning read.
- [ ] `pytest python/tests` and `ruff check python problems` pass.
- [ ] `CURRICULUM.md` totals, the module's `core` list, and the area file are current.
- [ ] The owner has the list of what was not verified.

## 7. What has gone wrong before

Defects, how they surfaced, and the rule that came out of each.

1. **Hidden cases leaked their purpose.** Hidden case ids were the test function names,
   and failure messages revealed expected sizes. Reported by the first authors and again
   in milestone 2. Now: opaque `hidden-N` ids, audited `check` messages.
2. **"Do not call the built-in" was not enforced.** Found after the first 12 problems.
   Now: `forbidden` lists and a shortcut mutant.
3. **Spec mutants that cannot fail.** Many curriculum mutants were mathematically
   identical to the correct solution or invisible through the signature (RoPE position
   shift, softmax after top-k, renormalize before min-p). Authors caught these because
   every mutant must be rejected. Rule: replace and report.
4. **A spec contradicted its sources.** `prompt-lookup-draft` followed the curriculum
   ("most recent occurrence") although `transformers` and vLLM take the earliest; the
   author reported it and the manager had it rewritten before the commit. The curriculum
   had also said min-p is on by default in mlx-lm (it is not) and described the house
   tokenizer and chat format wrongly. Rule: the source wins over the spec.
5. **An author's formatter rewrote other problems.** `ruff format` on a whole track
   directory reformatted code blocks in six Markdown files of five other problems; the
   changes were committed and the commit had to be amended. Rule: one path per problem, and
   `git status --short` before reporting and before committing.
6. **A citation with unverifiable authors.** An editorial named a paper's authors from
   memory; sources disagreed. The names were removed. Rule: title and arXiv id or venue,
   checked.
7. **Parallel authors collided.** Shared scratch files were overwritten, machine load
   produced false timing warnings, and the owner's usage limit was hit three times.
   Rule: one subagent at a time, each with its own scratch folder. (Later, one full
   validation under load showed 122 of 123 and the failing problem was never identified;
   every run since was clean.)
8. **Measured facts that held only sometimes.** A float16 figure for the real model was
   true only for some prompts; curriculum speedup estimates were optimistic. Rule:
   measure on this machine and size the case to the measurement.
9. **Kit bugs found by problems.** For example `SyncSpy` missing slice reads,
   `CompileCounter` reading zero under `fullgraph=True`, the Triton shim disagreeing with
   Triton's docs, `SimCluster` raising when read at the instant a step ends (found in the
   closing pass by the last capstone). Rule: authors report, the next kit engineer fixes
   with a regression test.
10. **Figures not seen in the app.** Authors rendered figures standalone; the manager
    checked only some in both themes. Rule: step 7.

## 8. Known gaps between this procedure and the record

- **No independent review** of any problem (step 6). The fairness and "fails for the
  right reason" checks rest on author reports.
- **`CURRICULUM.md` drifted.** Its header says to fix an ambiguous spec there first.
  Authors fixed the problem instead and reported; most corrections were never written
  back, so for those entries the problem's statement is right and the spec is stale.
- **"No warning" was not met everywhere.** The final catalog run carried ten timing
  warnings (slow JAX references).
- **Sources from memory remain** in some editorials (a few engine behaviors, paper
  venues, one softcap value); the reports name them.
- **Formatting.** `ruff check` is the gate. `ruff format` was asked of authors only from
  milestone 3, and the earlier tracks do not pass `ruff format --check`. Do not run the
  formatter across existing problems as part of a batch.
- **Repetition counts varied** (two, three, or four validations) with the batch.
- **Nothing runs automatically.** There is no CI and no pre-commit hook; the gate runs
  when someone runs it.
- **CUDA cases have never run**, and the Triton shim was never compared with real Triton.
