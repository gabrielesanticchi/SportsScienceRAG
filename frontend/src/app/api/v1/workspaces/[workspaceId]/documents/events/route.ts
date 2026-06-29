import { requireAuth } from "@/lib/mock-api/auth-middleware";
import { subscribeParsingEvents } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ workspaceId: string }>;
}

export async function GET(request: Request, { params }: RouteParams) {
  const unauthorized = requireAuth(request);
  if (unauthorized) return unauthorized;

  const { workspaceId } = await params;
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      const send = (data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: parsing_update\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      const unsubscribe = subscribeParsingEvents(workspaceId, send);

      const keepAlive = setInterval(() => {
        controller.enqueue(encoder.encode(": keepalive\n\n"));
      }, 15000);

      request.signal.addEventListener("abort", () => {
        clearInterval(keepAlive);
        unsubscribe();
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
