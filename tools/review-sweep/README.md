# Review sweep

`sweep.py` runs the checks of a problem review that a script can do, so that a
reviewer starts with leads. It changes no problem.

```sh
python3 tools/review-sweep/sweep.py --out <dir> [--modules a,b] [--ids x,y] [-j 3]
```

Without `--modules` it covers the modules of `areas/inference-and-serving.toml`.
It writes `<dir>/report.md`, `<dir>/results.json`, and one file per problem
under `<dir>/problems/`. Problems with a speed or service judge run one at a
time after the others.

For each problem it submits every mutant with `--reveal`, one mutant as a
learner sees it, and the reference, all in the oracle framework.

| flag | weight | meaning |
|---|---|---|
| `reference_not_accepted` | 5 | the reference failed a full submit |
| `mutant_not_rejected` | 5 | a mutant was accepted |
| `hidden_leak` | 3 to 4 | a hidden case showed its name, detail, or stdout to the learner |
| `mutant_did_not_run` | 3 | the run ended without cases |
| `crash_only` | 3 | a mutant is rejected only by an exception or a timeout, not by a wrong value |
| `nan_only` | 2 | every failing case is a NaN or an infinity and the `# BUG:` line does not say so |
| `hidden_only` | 2 | a mutant is caught only by hidden cases, so no public case shows its rule |
| `forbidden_source` | 2 | the problem bans words in the source; read them for ordinary words |
| `ban_not_enforced` | 2 | the statement rules something out and there is no `forbidden` list |
| `fails_every_case` | 1 | a mutant fails all cases, the worked example included |
| `no_mutation_mutant`, `no_tie_mutant` | 1 | the statement has such a rule and no mutant is about it |
| `hidden_message_values` | 1 | hidden failure messages state an expected value |
| `slow_case` | 1 | a reference case took more than 3 s |
| `single_case`, `idle_case`, `skipped_case` | 0 | recorded, not scored |

A flag is a lead. The statement-reading flags are word matches and are often
wrong. The sweep cannot see an honest solution that is rejected, a wrong
solution that no mutant resembles, a fact that contradicts its source, or an
ambiguous statement: those need a reviewer
([`docs/PROBLEM_PROCESS.md`](../../docs/PROBLEM_PROCESS.md), step 6).
