import { describe, expect, it, vi } from "vitest";
import { SessionTransport } from "./transport";

/** A send function whose calls are resolved or rejected by hand, in any order. */
function manualSender() {
  const calls: { message: string; resolve: () => void; reject: (error: Error) => void }[] = [];
  const send = (message: string) =>
    new Promise<void>((resolve, reject) => {
      calls.push({ message, resolve, reject });
    });
  return { calls, send };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("SessionTransport", () => {
  it("sends one message at a time, in the order they were queued", async () => {
    const { calls, send } = manualSender();
    const transport = new SessionTransport(send);
    transport.send("didChange");
    transport.send("completion");
    transport.send("hover");
    await settle();
    // The second message waits for the first: over HTTP it could otherwise overtake it.
    expect(calls.map((call) => call.message)).toEqual(["didChange"]);
    calls[0]?.resolve();
    await settle();
    expect(calls.map((call) => call.message)).toEqual(["didChange", "completion"]);
    calls[1]?.resolve();
    await settle();
    expect(calls.map((call) => call.message)).toEqual(["didChange", "completion", "hover"]);
  });

  it("treats a failed send as a lost message and keeps going", async () => {
    const { calls, send } = manualSender();
    const lost = vi.fn();
    const transport = new SessionTransport(send, lost);
    transport.send("first");
    transport.send("second");
    await settle();
    calls[0]?.reject(new Error("the editor session is not running"));
    await settle();
    expect(lost).toHaveBeenCalledTimes(1);
    expect(lost.mock.calls[0]?.[0]).toBeInstanceOf(Error);
    expect(calls.map((call) => call.message)).toEqual(["first", "second"]);
  });

  it("never throws from send, so a notification sent from a promise cannot reject unhandled", () => {
    const transport = new SessionTransport(() => {
      throw new Error("synchronous failure");
    });
    expect(() => transport.send("x")).not.toThrow();
    transport.close();
    expect(() => transport.send("y")).not.toThrow();
  });

  it("delivers server messages to every subscriber until it is unsubscribed", () => {
    const transport = new SessionTransport(async () => {});
    const first = vi.fn();
    const second = vi.fn();
    transport.subscribe(first);
    transport.subscribe(second);
    transport.deliver("a");
    transport.unsubscribe(first);
    transport.deliver("b");
    expect(first.mock.calls).toEqual([["a"]]);
    expect(second.mock.calls).toEqual([["a"], ["b"]]);
  });

  it("goes quiet when the session changes: nothing queued is sent, nothing is delivered", async () => {
    const { calls, send } = manualSender();
    const lost = vi.fn();
    const transport = new SessionTransport(send, lost);
    const handler = vi.fn();
    transport.subscribe(handler);
    transport.send("in flight");
    transport.send("queued");
    await settle();
    transport.close();
    expect(transport.closed).toBe(true);
    // The message in flight fails on the old session; that is not reported as a loss.
    calls[0]?.reject(new Error("the editor session is not running"));
    await settle();
    transport.send("after close");
    transport.deliver("late message from the old server");
    await settle();
    expect(calls.map((call) => call.message)).toEqual(["in flight"]);
    expect(handler).not.toHaveBeenCalled();
    expect(lost).not.toHaveBeenCalled();
  });
});
