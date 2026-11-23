-- Upgrade databases created from the earlier schema without dropping tenant data.
ALTER TABLE documents_registry
    ADD COLUMN IF NOT EXISTS sha256 char(64),
    ADD COLUMN IF NOT EXISTS version integer NOT NULL DEFAULT 1;

-- Existing records have no source-file digest. Keep those hashes unknown; new
-- ingestion writes always provide a real SHA-256 value.
ALTER TABLE parent_documents
    ADD COLUMN IF NOT EXISTS heading_path text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS page_start integer NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS is_table boolean NOT NULL DEFAULT false;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'documents_registry_sha256_check'
          AND conrelid = 'documents_registry'::regclass
    ) THEN
        ALTER TABLE documents_registry
            ADD CONSTRAINT documents_registry_sha256_check
            CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$');
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'documents_registry_version_check'
          AND conrelid = 'documents_registry'::regclass
    ) THEN
        ALTER TABLE documents_registry
            ADD CONSTRAINT documents_registry_version_check CHECK (version > 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'parent_documents_page_start_check'
          AND conrelid = 'parent_documents'::regclass
    ) THEN
        ALTER TABLE parent_documents
            ADD CONSTRAINT parent_documents_page_start_check CHECK (page_start > 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'documents_registry_tenant_source_version_key'
          AND conrelid = 'documents_registry'::regclass
    ) THEN
        WITH ranked AS (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY tenant_id, source_name
                       ORDER BY created_at, id
                   )::integer AS version
            FROM documents_registry
        )
        UPDATE documents_registry AS registry
        SET version = ranked.version
        FROM ranked
        WHERE registry.id = ranked.id;

        ALTER TABLE documents_registry
            ADD CONSTRAINT documents_registry_tenant_source_version_key
            UNIQUE (tenant_id, source_name, version);
    END IF;
END;
$$;

CREATE UNIQUE INDEX IF NOT EXISTS documents_registry_active_hash_idx
    ON documents_registry (tenant_id, sha256)
    WHERE status = 'active' AND sha256 IS NOT NULL;

CREATE OR REPLACE FUNCTION efm_private.row_access_allowed(
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
