import { describe, expect, it } from "vitest";
import type { EnvStatus, ModelAsset, TargetInfo } from "@/lib/api";
import { parseSettingsSection, routes } from "@/lib/routes";
import { settingsAttention } from "./settings";

const env: EnvStatus = {
  uv: "0.13.8",
  python: "3.15.2",
  root: "/r",
  frameworks: [
    { framework: "torch", installed: true, version: "2.15.0" },
    { framework: "mlx", installed: false, version: null },
  ],
};

const asset = (name: string, state: ModelAsset["state"]): ModelAsset => ({
  name,
  repo: `org/${name}`,
  revision: "abc",
  files: [],
  sizeMb: 700,
  state,
  reason: null,
  path: `/models/${name}`,
  bytesOnDisk: 0,
  problems: [],
});

const target = (name: string, ok: boolean | null): TargetInfo => ({
  name,
  kind: name === "local" ? "local" : "ssh",
  label: name,
  host: null,
  remoteDir: null,
  lastCheck: ok === null ? null : { ok, checkedAt: "2026-11-01T00:00:00Z", steps: [], doctor: null },
});

const marked = (facts: Parameters<typeof settingsAttention>[0]) => [...settingsAttention(facts)];

describe("settingsAttention", () => {
  it("marks nothing while everything is unknown or in order", () => {
    expect(marked({})).toEqual([]);
    expect(
      marked({
        env,
        assets: [asset("a", "present"), asset("b", "missing")],
        targets: [target("local", true), target("ssh:new", null)],
        issues: [],
      }),
    ).toEqual([]);
  });

  it("marks the environment when a solution cannot run or an install failed", () => {
    expect(marked({ env: { ...env, uv: null } })).toEqual(["environment"]);
    expect(marked({ env: { ...env, python: null } })).toEqual(["environment"]);
    expect(marked({ env, installFailed: true })).toEqual(["environment"]);
  });

  it("does not mark the environment for an optional framework that is not installed", () => {
    expect(marked({ env })).toEqual([]);
  });

  it("marks models for an incomplete or failed download, not for one under way", () => {
    expect(marked({ assets: [asset("a", "incomplete")] })).toEqual(["models"]);
    expect(
      marked({ assets: [asset("a", "missing")], downloads: { a: { phase: "failed", message: "reset" } } }),
    ).toEqual(["models"]);
    expect(
      marked({
        assets: [asset("a", "incomplete")],
        downloads: { a: { phase: "downloading", doneBytes: 1, totalBytes: 2, file: null } },
      }),
    ).toEqual([]);
    expect(marked({ assets: [asset("a", "incomplete")], downloads: { a: { phase: "finished" } } })).toEqual(
      [],
    );
  });

  it("marks run targets when a last check failed", () => {
    expect(marked({ targets: [target("local", true), target("ssh:studio", false)] })).toEqual(["targets"]);
  });

  it("marks files when the catalog skipped entries", () => {
    expect(marked({ issues: [{ path: "/r/problems/x", message: "no statement" }] })).toEqual(["files"]);
  });
});

describe("the settings section in the URL", () => {
  it("falls back to General for anything unknown", () => {
    expect(parseSettingsSection("models")).toBe("models");
    expect(parseSettingsSection("editor")).toBe("general");
    expect(parseSettingsSection(undefined)).toBe("general");
    expect(parseSettingsSection(3)).toBe("general");
  });

  it("leaves General out of the link", () => {
    expect(routes.settings()).toEqual({ to: "/settings", search: {} });
    expect(routes.settings("general")).toEqual({ to: "/settings", search: {} });
    expect(routes.settings("targets")).toEqual({ to: "/settings", search: { section: "targets" } });
  });
});
