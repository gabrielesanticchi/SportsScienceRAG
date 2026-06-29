-- PostgreSQL bootstrap: extensions, tables, RLS policies, seed tenant

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TYPE document_status AS ENUM (
    'PENDING',
    'PROCESSING',
    'COMPLETED',
    'FAILED'
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'application/pdf',
    sha256 TEXT NOT NULL,
    s3_key TEXT NOT NULL,
    upload_token TEXT UNIQUE,
    status document_status NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    page_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, sha256)
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    point_id TEXT NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index),
    UNIQUE (point_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_sha256 ON documents (tenant_id, sha256);
CREATE INDEX IF NOT EXISTS idx_documents_tenant_status ON documents (tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON document_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks (document_id);

-- Row-Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY documents_tenant_isolation ON documents
    USING (tenant_id::text = current_setting('app.current_tenant', true));

CREATE POLICY chunks_tenant_isolation ON document_chunks
    USING (tenant_id::text = current_setting('app.current_tenant', true));

CREATE POLICY users_tenant_isolation ON users
    USING (tenant_id::text = current_setting('app.current_tenant', true));

ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;

-- Application role for RLS-aware connections
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ingest_app') THEN
        CREATE ROLE ingest_app LOGIN PASSWORD 'ingest_secret';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO ingest_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ingest_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ingest_app;

-- Seed demo tenant for local development
INSERT INTO tenants (id, name)
VALUES ('11111111-1111-1111-1111-111111111111', 'demo-tenant')
ON CONFLICT (name) DO NOTHING;

INSERT INTO users (id, tenant_id, email, role)
VALUES (
    '22222222-2222-2222-2222-222222222222',
    '11111111-1111-1111-1111-111111111111',
    'demo@example.com',
    'admin'
)
ON CONFLICT (email) DO NOTHING;
