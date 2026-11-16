import { describe, expect, it } from "vitest";
import type { EditorStatus } from "@/lib/api/types";
import type { EditorConnection } from "./session";
import { suggestionsState, suggestionsVersion } from "./status";
import { SessionTransport } from "./transport";

const STATUS: EditorStatus = {
  enabled: true,
  installed: true,
  installing: false,
  serverVersion: "0.0.104",
  state: null,
};
const connection: EditorConnection = {
  sessionId: "s",
  generation: 1,
  rootUri: "file:///ws",
  serverName: "ty",
  serverVersion: "0.0.105",
  transport: new SessionTransport(async () => {}),
};

describe("suggestionsState", () => {
  it("puts the preference first, then whether the server is installed", () => {
    expect(suggestionsState({ ...STATUS, enabled: false }, { status: "ready", connection })).toEqual({
      kind: "off",
    });
    expect(suggestionsState({ ...STATUS, installed: false }, { status: "idle" })).toEqual({
      kind: "notInstalled",
    });
  });

  it("follows the live session once the server is installed", () => {
    expect(suggestionsState(STATUS, { status: "idle" })).toEqual({ kind: "installed" });
    expect(suggestionsState(STATUS, { status: "starting" })).toEqual({ kind: "starting" });
    expect(suggestionsState(STATUS, { status: "ready", connection })).toEqual({ kind: "running" });
    expect(
      suggestionsState(STATUS, { status: "restarting", attempt: 1, maxAttempts: 4, message: "m" }),
    ).toEqual({ kind: "restarting" });
    expect(suggestionsState(STATUS, { status: "failed", message: "It keeps stopping." })).toEqual({
      kind: "failed",
      message: "It keeps stopping.",
    });
  });

  it("prefers the running server's version", () => {
    expect(suggestionsVersion(STATUS, { status: "idle" })).toBe("0.0.104");
    expect(suggestionsVersion(STATUS, { status: "ready", connection })).toBe("0.0.105");
    expect(suggestionsVersion({ ...STATUS, serverVersion: null }, { status: "idle" })).toBeNull();
  });
});
