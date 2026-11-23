/**
 * An incremental parser for server-sent events (the WHATWG "event stream" format), for a
 * stream read with fetch: EventSource cannot POST, and `POST /chat` streams its answer.
 *
 *     const parser = new SseParser();
 *     for (const event of parser.feed(chunk)) handle(event);
 *
 * Chunks may split anywhere (inside a line, between `\r` and `\n`). Lines end with `\n`,
 * `\r\n` or `\r`; a blank line dispatches the event; `data:` lines are joined with `\n`;
 * `:` lines are comments; an event with no data is dropped, as is an unfinished one when
 * the stream ends.
 */

export type SseEvent = {
  /** `message` when the event has no `event:` line. */
  event: string;
  data: string;
  /** The last `id:` seen on the stream, if any. */
  id?: string;
};

export class SseParser {
  private buffer = "";
  private started = false;
  private skipLf = false; // the last chunk ended with \r: a leading \n belongs to it
  private type = "";
  private data: string[] = [];
  private lastId: string | undefined;

  /** The events completed by ``chunk``. */
  feed(chunk: string): SseEvent[] {
    let text = chunk;
    if (!this.started && text.length > 0) {
      this.started = true;
      if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // a leading BOM is ignored
    }
    if (this.skipLf && text.startsWith("\n")) text = text.slice(1);
    this.skipLf = false;
    this.buffer += text;

    const out: SseEvent[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const c = this.buffer[i];
      if (c !== "\n" && c !== "\r") continue;
      this.line(this.buffer.slice(start, i), out);
      if (c === "\r") {
        if (i + 1 < this.buffer.length) {
          if (this.buffer[i + 1] === "\n") i++;
        } else {
          this.skipLf = true;
        }
      }
      start = i + 1;
    }
    this.buffer = this.buffer.slice(start);
    return out;
  }

  /** Call when the stream ends: an event without its closing blank line is discarded. */
  end(): SseEvent[] {
    this.buffer = "";
    this.type = "";
    this.data = [];
    return [];
  }

  private line(line: string, out: SseEvent[]): void {
    if (line === "") {
      if (this.data.length > 0) {
        const event: SseEvent = { event: this.type || "message", data: this.data.join("\n") };
        if (this.lastId !== undefined) event.id = this.lastId;
        out.push(event);
      }
      this.type = "";
      this.data = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.type = value;
    else if (field === "data") this.data.push(value);
    else if (field === "id" && !value.includes("\0")) this.lastId = value;
    // `retry` and unknown fields are ignored
  }
}
