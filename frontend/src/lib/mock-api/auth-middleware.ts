import { NextResponse } from "next/server";

export function requireAuth(request: Request): NextResponse | null {
  const auth = request.headers.get("Authorization");
  const url = new URL(request.url);
  const queryToken = url.searchParams.get("token");

  if (auth?.startsWith("Bearer ") || queryToken) {
    return null;
  }

  return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
}
