type Handler = (message: string) => void;

/**
 * The transport an LSP client talks through for one server process: bare
 * JSON-RPC text in both directions (what `@codemirror/lsp-client` expects of a
 * `Transport`).
 *
 * Messages are handed to `sendMessage` one at a time and in order: over HTTP
 * two requests in flight can overtake each other, and a language server must
 * see `didChange` before the completion request that depends on it. A send
 * that fails is a lost message (`onSendError` hears about it); the ones behind
 * it still go out. After `close` nothing is sent and nothing is delivered.
 */
export class SessionTransport {
  private readonly handlers: Handler[] = [];
  private queue: Promise<void> = Promise.resolve();
  private open = true;

  constructor(
    private readonly sendMessage: (message: string) => Promise<void>,
    private readonly onSendError: (error: Error) => void = () => {},
  ) {}

  get closed(): boolean {
    return !this.open;
  }

  /**
   * Queues a message for the server. Never throws: the client sends
   * notifications from promise callbacks, where a throw would surface as an
   * unhandled rejection, and a request that is never answered times out there.
   */
  send(message: string): void {
    if (!this.open) return;
    this.queue = this.queue.then(async () => {
      if (!this.open) return;
      try {
        await this.sendMessage(message);
      } catch (cause) {
        if (this.open) this.onSendError(cause instanceof Error ? cause : new Error(String(cause)));
      }
    });
  }

  subscribe(handler: Handler): void {
    this.handlers.push(handler);
  }

  unsubscribe(handler: Handler): void {
    const index = this.handlers.indexOf(handler);
    if (index >= 0) this.handlers.splice(index, 1);
  }

  /** Hands one message from the server to the subscribers. */
  deliver(message: string): void {
    if (!this.open) return;
    for (const handler of [...this.handlers]) handler(message);
  }

  /** Ends the transport: its server process is gone. Queued messages are dropped. */
  close(): void {
    this.open = false;
    this.handlers.length = 0;
  }
}
