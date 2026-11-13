import { type NavigateOptions, useNavigate } from "@tanstack/react-router";
import { useCallback } from "react";
import type { RouteTarget } from "@/lib/routes";

/** Imperative navigation to a target from lib/routes. */
export function useGo(): (target: RouteTarget, options?: { replace?: boolean }) => void {
  const navigate = useNavigate();
  return useCallback(
    (target, options) => {
      void navigate({ ...target, ...options } as unknown as NavigateOptions);
    },
    [navigate],
  );
}
