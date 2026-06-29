import { NextResponse } from "next/server";

import { requireAuth } from "@/lib/mock-api/auth-middleware";
import { getDocument } from "@/lib/mock-api/store";

export async function POST(request: Request) {
  const unauthorized = requireAuth(request);
  if (unauthorized) return unauthorized;

  const body = (await request.json()) as {
    query: string;
    workspace_id: string;
    document_ids: string[];
  };

  if (!body.document_ids.length) {
    return NextResponse.json(
      { error: "At least one document must be selected" },
      { status: 400 },
    );
  }

  const citedDocs = body.document_ids
    .map((id) => getDocument(body.workspace_id, id))
    .filter(Boolean);

  const firstDoc = citedDocs[0];
  const firstBox = firstDoc?.boundingBoxes[0];

  const citations = citedDocs.slice(0, 2).map((doc, index) => ({
    id: crypto.randomUUID(),
    documentId: doc!.id,
    boundingBoxId: doc!.boundingBoxes[index]?.id ?? doc!.boundingBoxes[0]?.id ?? "",
    text: doc!.boundingBoxes[index]?.text ?? "Referenced passage",
    page: doc!.boundingBoxes[index]?.page ?? 1,
    label: String(index + 1),
  }));

  const answer = `Based on ${body.document_ids.length} selected source(s), here is a synthesized answer to: "${body.query}". ${
    firstBox
      ? `The most relevant passage appears on page ${firstBox.page} (${firstBox.type}).`
      : "Upload and process documents to enable citation-backed answers."
  }`;

  return NextResponse.json({ answer, citations });
}
