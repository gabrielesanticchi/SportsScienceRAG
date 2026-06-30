import { NextResponse } from "next/server";

import { requireAuth } from "@/lib/mock-api/auth-middleware";
import { createDocument } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ workspaceId: string }>;
}

export async function POST(request: Request, { params }: RouteParams) {
  const unauthorized = requireAuth(request);
  if (unauthorized) return unauthorized;

  const { workspaceId } = await params;
  const body = (await request.json()) as {
    filename: string;
    size_bytes: number;
  };

  const { documentId, uploadToken } = createDocument(
    workspaceId,
    body.filename,
  );

  const origin = new URL(request.url).origin;

  return NextResponse.json({
    document_id: documentId,
    upload_url: `${origin}/api/v1/upload/${uploadToken}`,
  });
}
