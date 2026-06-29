import { NextResponse } from "next/server";

import { completeUpload } from "@/lib/mock-api/store";

interface RouteParams {
  params: Promise<{ uploadToken: string }>;
}

export async function PUT(request: Request, { params }: RouteParams) {
  const { uploadToken } = await params;
  const buffer = await request.arrayBuffer();

  const success = completeUpload(uploadToken, buffer);
  if (!success) {
    return NextResponse.json({ error: "Invalid upload token" }, { status: 404 });
  }

  return new NextResponse(null, { status: 200 });
}
