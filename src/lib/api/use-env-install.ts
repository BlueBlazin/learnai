import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { backend } from "./index";
import { keys } from "./queries";
import type { Framework } from "./types";

export interface InstallState {
  framework: Framework | null;
  status: "idle" | "running" | "succeeded" | "failed";
  log: string[];
  message: string | null;
}

const idle: InstallState = { framework: null, status: "idle", log: [], message: null };

/** Runs a framework install and collects its streamed log. One install at a time. */
export function useEnvInstall() {
  const client = useQueryClient();
  const [state, setState] = useState<InstallState>(idle);

  const install = useCallback(
    async (framework: Framework) => {
      setState({ framework, status: "running", log: [], message: null });
      try {
        await backend.envInstall(framework, (event) => {
          if (event.type === "log") {
            setState((s) => ({ ...s, log: [...s.log, event.text] }));
          } else {
            setState((s) => ({
              ...s,
              status: event.ok ? "succeeded" : "failed",
              message: event.message,
            }));
            void client.invalidateQueries({ queryKey: keys.env });
          }
        });
      } catch (error) {
        setState((s) => ({
          ...s,
          status: "failed",
          message: error instanceof Error ? error.message : String(error),
        }));
      }
    },
    [client],
  );

  return { state, install };
}
