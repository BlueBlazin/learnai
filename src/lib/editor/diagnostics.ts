/** The parts of an LSP diagnostic the editor uses. */
export interface ServerDiagnostic {
  range: { start: { line: number; character: number }; end: { line: number; character: number } };
  /** 1 error, 2 warning, 3 information, 4 hint. Absent means error. */
  severity?: number;
  message: string;
  code?: string | number;
}

export type DiagnosticSeverity = "error" | "warning" | "info";

const HINT = 4;

/**
 * The diagnostics worth underlining. Hints (severity 4: an unused variable, an
 * unreachable branch) are dropped: a half-written solution is full of them and
 * none is a mistake.
 */
export function visibleDiagnostics<T extends ServerDiagnostic>(diagnostics: readonly T[]): T[] {
  return diagnostics.filter((diagnostic) => (diagnostic.severity ?? 1) !== HINT);
}

export function severityOf(diagnostic: ServerDiagnostic): DiagnosticSeverity {
  const severity = diagnostic.severity ?? 1;
  return severity === 1 ? "error" : severity === 2 ? "warning" : "info";
}
