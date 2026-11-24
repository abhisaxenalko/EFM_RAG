\getenv efm_app_password EFM_APP_PASSWORD
\getenv efm_database POSTGRES_DB

SELECT 'CREATE ROLE efm_owner NOLOGIN'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'efm_owner')
\gexec

SELECT format(
    'CREATE ROLE efm_app LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD %L',
    :'efm_app_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'efm_app')
\gexec

SELECT format(
    'ALTER ROLE efm_app LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD %L',
    :'efm_app_password'
)
\gexec

CREATE EXTENSION IF NOT EXISTS vector;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE :"efm_database" TO efm_app;
GRANT USAGE ON SCHEMA public TO efm_app;

-- Tenant identity comes from a tenant-specific database login, not a caller-settable GUC.
CREATE SCHEMA efm_private AUTHORIZATION efm_owner;
REVOKE ALL ON SCHEMA efm_private FROM PUBLIC;
GRANT USAGE ON SCHEMA efm_private TO efm_app;

CREATE TABLE efm_private.tenant_principals (
    principal_name name PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    allowed_roles text[] NOT NULL
);
ALTER TABLE efm_private.tenant_principals OWNER TO efm_owner;
REVOKE ALL ON efm_private.tenant_principals FROM PUBLIC, efm_app;

CREATE FUNCTION efm_private.row_access_allowed(
    row_tenant_id varchar,
    row_access_role text
)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, efm_private
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM efm_private.tenant_principals AS principal
        WHERE principal.principal_name = session_user
          AND principal.tenant_id = row_tenant_id
          AND row_access_role = ANY (principal.allowed_roles)
          AND row_tenant_id = current_setting('app.current_tenant_id', true)
          AND row_access_role = ANY (
              string_to_array(current_setting('app.allowed_roles', true), ',')
          )
    );
$$;
ALTER FUNCTION efm_private.row_access_allowed(varchar, text) OWNER TO efm_owner;
REVOKE ALL ON FUNCTION efm_private.row_access_allowed(varchar, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION efm_private.row_access_allowed(varchar, text) TO efm_app;

CREATE TABLE documents_registry (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    access_role text NOT NULL DEFAULT 'all',
    status text NOT NULL DEFAULT 'active',
    source_name text NOT NULL,
    sha256 char(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    version integer NOT NULL CHECK (version > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, id),
    CONSTRAINT documents_registry_tenant_source_version_key
        UNIQUE (tenant_id, source_name, version)
);
CREATE UNIQUE INDEX documents_registry_active_hash_idx
    ON documents_registry (tenant_id, sha256)
    WHERE status = 'active';

CREATE TABLE parent_documents (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    access_role text NOT NULL DEFAULT 'all',
    status text NOT NULL DEFAULT 'active',
    registry_id bigint NOT NULL,
    heading_path text NOT NULL DEFAULT '',
    page_start integer NOT NULL DEFAULT 1 CHECK (page_start > 0),
    is_table boolean NOT NULL DEFAULT false,
    content text NOT NULL,
    UNIQUE (tenant_id, id),
    FOREIGN KEY (tenant_id, registry_id)
        REFERENCES documents_registry (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE document_chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id varchar(64) NOT NULL,
    access_role text NOT NULL DEFAULT 'all',
    status text NOT NULL DEFAULT 'active',
    parent_document_id bigint NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    suspicious boolean,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    embedding vector(1024) NOT NULL,
    FOREIGN KEY (tenant_id, parent_document_id)
        REFERENCES parent_documents (tenant_id, id) ON DELETE CASCADE
);

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS suspicious boolean;
ALTER TABLE document_chunks
    ALTER COLUMN suspicious DROP NOT NULL,
    ALTER COLUMN suspicious DROP DEFAULT;

CREATE INDEX documents_registry_tenant_status_idx
    ON documents_registry (tenant_id, status);
CREATE INDEX parent_documents_tenant_status_idx
    ON parent_documents (tenant_id, status);
CREATE INDEX document_chunks_tenant_status_idx
    ON document_chunks (tenant_id, status);
CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX document_chunks_tsv_gin_idx
    ON document_chunks USING gin (tsv);

ALTER TABLE documents_registry OWNER TO efm_owner;
ALTER TABLE parent_documents OWNER TO efm_owner;
ALTER TABLE document_chunks OWNER TO efm_owner;

ALTER TABLE documents_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents_registry FORCE ROW LEVEL SECURITY;
ALTER TABLE parent_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE parent_documents FORCE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;

CREATE POLICY documents_registry_tenant_access ON documents_registry
    FOR ALL TO efm_app
    USING (efm_private.row_access_allowed(tenant_id, access_role))
    WITH CHECK (efm_private.row_access_allowed(tenant_id, access_role));
CREATE POLICY documents_registry_owner_maintenance ON documents_registry
    FOR ALL TO efm_owner USING (true) WITH CHECK (true);

CREATE POLICY parent_documents_tenant_access ON parent_documents
    FOR ALL TO efm_app
    USING (efm_private.row_access_allowed(tenant_id, access_role))
    WITH CHECK (efm_private.row_access_allowed(tenant_id, access_role));
CREATE POLICY parent_documents_owner_maintenance ON parent_documents
    FOR ALL TO efm_owner USING (true) WITH CHECK (true);

CREATE POLICY document_chunks_tenant_access ON document_chunks
    FOR ALL TO efm_app
    USING (efm_private.row_access_allowed(tenant_id, access_role))
    WITH CHECK (efm_private.row_access_allowed(tenant_id, access_role));
CREATE POLICY document_chunks_owner_maintenance ON document_chunks
    FOR ALL TO efm_owner USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON documents_registry, parent_documents, document_chunks TO efm_app;
GRANT USAGE, SELECT ON SEQUENCE
    documents_registry_id_seq,
    parent_documents_id_seq,
    document_chunks_id_seq
TO efm_app;

-- Provision one LOGIN role per tenant; map that login to its tenant and allowed roles.
-- Then GRANT efm_app TO tenant_login so it inherits table access and these RLS policies.
-- Do not add the shared efm_app login to tenant_principals.
