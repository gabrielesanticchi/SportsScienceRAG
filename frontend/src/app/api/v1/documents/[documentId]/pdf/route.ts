import { NextResponse } from "next/server";

import { getDocumentPdf } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ documentId: string }>;
}

export async function GET(_request: Request, { params }: RouteParams) {
  const { documentId } = await params;
  const buffer = getDocumentPdf(documentId);

  if (!buffer) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return new NextResponse(buffer, {
    headers: {
      "Content-Type": "application/pdf",
      "Cache-Control": "private, max-age=3600",
    },
  });
}
