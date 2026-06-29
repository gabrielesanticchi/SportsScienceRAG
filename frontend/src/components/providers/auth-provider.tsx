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

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);
  const [tenantId, setTenantIdState] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const tenant = process.env.NEXT_PUBLIC_DEMO_TENANT_ID ?? "tenant-demo";
    let existing = getAuthToken();

    if (!existing) {
      existing = buildDemoJwt(tenant);
      setAuthToken(existing);
      setTenantId(tenant);
    }

    setTokenState(existing);
    setTenantIdState(tenant);
    setIsReady(true);
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
