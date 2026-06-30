"use client";

import { createContext, useContext, useEffect, useState } from "react";

import { getAuthToken, setAuthToken, setTenantId } from "@/lib/auth";

interface AuthContextValue {
  token: string | null;
  tenantId: string | null;
  isReady: boolean;
}

const AuthContext = createContext<AuthContextValue>({
  token: null,
  tenantId: null,
  isReady: false,
});

function buildDemoJwt(tenantId: string): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const payload = btoa(
    JSON.stringify({
      sub: "demo-user",
      tenant_id: tenantId,
      exp: Math.floor(Date.now() / 1000) + 86400,
    }),
  );
  return `${header}.${payload}.demo-signature`;
}

async function fetchBackendDevToken(apiBase: string): Promise<{ token: string; tenant_id: string } | null> {
  try {
    const response = await fetch(`${apiBase}/auth/dev-token`);
    if (!response.ok) return null;
    return (await response.json()) as { token: string; tenant_id: string };
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);
  const [tenantId, setTenantIdState] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "/api/v1";
    const tenant = process.env.NEXT_PUBLIC_DEMO_TENANT_ID ?? "tenant-demo";

    void (async () => {
      let existing = getAuthToken();
      const usesExternalBackend = apiBase.startsWith("http");

      if (usesExternalBackend) {
        const backendAuth = await fetchBackendDevToken(apiBase);
        if (backendAuth) {
          existing = backendAuth.token;
          setAuthToken(existing);
          setTenantId(backendAuth.tenant_id);
          setTokenState(existing);
          setTenantIdState(backendAuth.tenant_id);
          setIsReady(true);
          return;
        }
      }

      if (!existing) {
        existing = buildDemoJwt(tenant);
        setAuthToken(existing);
        setTenantId(tenant);
      }

      setTokenState(existing);
      setTenantIdState(tenant);
      setIsReady(true);
    })();
  }, []);

  return (
    <AuthContext.Provider value={{ token, tenantId, isReady }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
