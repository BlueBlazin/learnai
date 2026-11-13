import type { Backend } from "./backend";

/**
 * A backend whose implementation arrives later (a dynamic import). Every
 * `Backend` method already returns a promise, so each call simply waits for the
 * implementation first. This keeps heavy or development-only backends out of
 * the main bundle.
 */
export function lazyBackend(kind: Backend["kind"], load: () => Promise<Backend>): Backend {
  let loaded: Promise<Backend> | null = null;
  const inner = () => {
    loaded ??= load();
    return loaded;
  };
  return new Proxy({ kind } as Backend, {
    get(target, property) {
      if (property === "kind") return target.kind;
      if (typeof property !== "string" || property === "then") return undefined;
      return async (...args: unknown[]) => {
        const backend = (await inner()) as unknown as Record<string, (...args: unknown[]) => unknown>;
        const method = backend[property];
        if (typeof method !== "function") throw new Error(`The backend has no method "${property}".`);
        return method.apply(backend, args);
      };
    },
  });
}

/** A backend for builds that ship no data source: every call fails with `reason`. */
export function unavailableBackend(kind: Backend["kind"], reason: string): Backend {
  return lazyBackend(kind, () => Promise.reject(new Error(reason)));
}
