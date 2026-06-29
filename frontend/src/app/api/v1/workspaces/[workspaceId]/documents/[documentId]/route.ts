import { NextResponse } from "next/server";

import { requireAuth } from "@/lib/mock-api/auth-middleware";
import { getDocument } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ workspaceId: string; documentId: string }>;
}

export async function GET(request: Request, { params }: RouteParams) {
  const unauthorized = requireAuth(request);
  if (unauthorized) return unauthorized;

  const { workspaceId, documentId } = await params;
  const document = getDocument(workspaceId, documentId);

  if (!document) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json(document);
}
