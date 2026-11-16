import { describe, expect, it } from "vitest";
import { documentKey, documentUri } from "./uri";

const ROOT = "file:///Users/ada/learnai/workspace";

describe("documentUri", () => {
  it("forms the saved solution's URI from the root URI", () => {
    expect(documentUri(ROOT, { problemId: "stable-softmax", framework: "torch" })).toBe(
      "file:///Users/ada/learnai/workspace/stable-softmax/torch.py",
    );
    expect(documentUri(ROOT, { problemId: "rope", framework: "python", review: false })).toBe(
      "file:///Users/ada/learnai/workspace/rope/python.py",
    );
  });

  it("puts a review attempt under review/", () => {
    expect(documentUri(ROOT, { problemId: "stable-softmax", framework: "mlx", review: true })).toBe(
      "file:///Users/ada/learnai/workspace/stable-softmax/review/mlx.py",
    );
  });

  it("percent-encodes each segment and tolerates a trailing slash on the root", () => {
    expect(documentUri(`${ROOT}/`, { problemId: "a b/c#d", framework: "jax" })).toBe(
      "file:///Users/ada/learnai/workspace/a%20b%2Fc%23d/jax.py",
    );
  });
});

describe("documentKey", () => {
  it("tells documents apart by problem, framework, and review", () => {
    const keys = [
      documentKey({ problemId: "a", framework: "torch" }),
      documentKey({ problemId: "a", framework: "jax" }),
      documentKey({ problemId: "a", framework: "torch", review: true }),
      documentKey({ problemId: "b", framework: "torch" }),
    ];
    expect(new Set(keys).size).toBe(4);
    expect(documentKey(undefined)).toBeNull();
  });
});
