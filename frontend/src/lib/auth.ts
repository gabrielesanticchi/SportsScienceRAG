const AUTH_TOKEN_KEY = "rag_auth_token";
const TENANT_ID_KEY = "rag_tenant_id";

export function getAuthToken(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY);
}

export function getTenantIdFromToken(token: string): string | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const decoded = JSON.parse(atob(payload)) as { tenant_id?: string };
    return decoded.tenant_id ?? null;
  } catch {
    return null;
  }
}

export function getActiveTenantId(): string | null {
  const token = getAuthToken();
  if (!token) return localStorage.getItem(TENANT_ID_KEY);
  return getTenantIdFromToken(token);
}

export function setTenantId(tenantId: string): void {
  localStorage.setItem(TENANT_ID_KEY, tenantId);
}
