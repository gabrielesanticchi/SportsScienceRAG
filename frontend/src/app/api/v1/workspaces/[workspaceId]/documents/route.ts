import { NextResponse } from "next/server";

import { requireAuth } from "@/lib/mock-api/auth-middleware";
import { listDocuments } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ workspaceId: string }>;
}

export async function GET(request: Request, { params }: RouteParams) {
  const unauthorized = requireAuth(request);
  if (unauthorized) return unauthorized;

  const { workspaceId } = await params;
  return NextResponse.json({ documents: listDocuments(workspaceId) });
}
