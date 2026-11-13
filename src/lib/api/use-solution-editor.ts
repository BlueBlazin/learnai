import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { backend } from "./index";
import { keys, useSolution } from "./queries";
import type { Framework, SolutionFile } from "./types";

export type SaveState = "saved" | "unsaved" | "saving" | "error";

const AUTOSAVE_DELAY_MS = 600;

interface PendingWrite {
  problemId: string;
  framework: Framework;
  code: string;
  timer: ReturnType<typeof setTimeout>;
}

export interface SolutionEditor {
  /** Undefined until the file has been read. */
  code: string | undefined;
  path: string | undefined;
  isStarter: boolean;
  loading: boolean;
  error: Error | null;
  saveState: SaveState;
  setCode: (code: string) => void;
  /** Writes any pending edit now. Await this before starting a run. */
  flush: () => Promise<void>;
  reset: () => Promise<void>;
}

/**
 * The editor's document for one problem and framework: reads the solution
 * file, autosaves edits through `writeSolution` after a short pause, and adopts
 * the file from disk when the window regains focus and nothing is unsaved.
 */
export function useSolutionEditor(problemId: string, framework: Framework | null): SolutionEditor {
  const client = useQueryClient();
  const query = useSolution(problemId, framework);
  const slot = `${problemId}/${framework ?? ""}`;
  const [draft, setDraft] = useState<{ slot: string; code: string } | null>(null);
  const [saveState, setSaveState] = useState<SaveState>("saved");
  const pending = useRef<PendingWrite | null>(null);
  const writing = useRef(0);

  const flush = useCallback(async () => {
    const write = pending.current;
    if (!write) return;
    clearTimeout(write.timer);
    pending.current = null;
    const queryKey = keys.solution(write.problemId, write.framework);
    const wasStarter = client.getQueryData<SolutionFile>(queryKey)?.isStarter ?? false;
    writing.current += 1;
    setSaveState("saving");
    try {
      await backend.writeSolution(write.problemId, write.framework, write.code);
      client.setQueryData<SolutionFile>(queryKey, (old) =>
        old ? { ...old, code: write.code, isStarter: false } : old,
      );
      if (wasStarter) {
        // The first edit moves the problem from "todo" to "attempted".
        void client.invalidateQueries({ queryKey: keys.problems });
        void client.invalidateQueries({ queryKey: keys.problem(write.problemId) });
      }
      if (!pending.current) setSaveState("saved");
    } catch {
      setSaveState("error");
    } finally {
      writing.current -= 1;
    }
  }, [client]);

  const setCode = useCallback(
    (code: string) => {
      if (framework === null) return;
      setDraft({ slot, code });
      if (pending.current) clearTimeout(pending.current.timer);
      pending.current = {
        problemId,
        framework,
        code,
        timer: setTimeout(() => void flush(), AUTOSAVE_DELAY_MS),
      };
      setSaveState("unsaved");
    },
    [flush, framework, problemId, slot],
  );

  // Leaving this problem or framework (or closing the workspace) saves first.
  // biome-ignore lint/correctness/useExhaustiveDependencies: slot is the trigger.
  useEffect(() => {
    setSaveState("saved");
    return () => void flush();
  }, [slot, flush]);

  // Adopt the file from disk when it changed underneath us and we have nothing unsaved.
  const diskCode = query.data?.code;
  useEffect(() => {
    if (diskCode === undefined || pending.current || writing.current > 0) return;
    setDraft((current) => (current && current.slot === slot && current.code !== diskCode ? null : current));
  }, [diskCode, slot]);

  const reset = useCallback(async () => {
    if (framework === null) return;
    if (pending.current) clearTimeout(pending.current.timer);
    pending.current = null;
    const file = await backend.resetSolution(problemId, framework);
    client.setQueryData(keys.solution(problemId, framework), file);
    setDraft(null);
    setSaveState("saved");
  }, [client, framework, problemId]);

  const code = draft && draft.slot === slot ? draft.code : diskCode;

  return {
    code,
    path: query.data?.path,
    isStarter: query.data ? query.data.isStarter && code === query.data.code : true,
    loading: query.isPending && framework !== null,
    error: query.error,
    saveState,
    setCode,
    flush,
    reset,
  };
}
