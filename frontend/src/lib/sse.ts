import { getAuthToken } from "@/lib/auth";
import type { ParsingEvent } from "@/lib/types";

export type ParsingEventHandler = (event: ParsingEvent) => void;

export function subscribeToParsingEvents(
  url: string,
  onEvent: ParsingEventHandler,
  onError?: (error: Event) => void,
): () => void {
  const token = getAuthToken();
  const eventSourceUrl = token
    ? `${url}?token=${encodeURIComponent(token)}`
    : url;

  const source = new EventSource(eventSourceUrl);

  source.addEventListener("parsing_update", (event) => {
    try {
      const data = JSON.parse(event.data) as ParsingEvent;
      onEvent(data);
    } catch {
      // ignore malformed events
    }
  });

  source.onerror = (error) => {
    onError?.(error);
  };

  return () => {
    source.close();
  };
}
