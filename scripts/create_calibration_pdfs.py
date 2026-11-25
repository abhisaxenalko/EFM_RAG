from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import get_settings
from app.ingestion.pipeline import ingest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "calibration"
DOCUMENTS = {
    "tenant_alpha": (
        "tenant_alpha_chiller_manual.pdf",
        "Tenant Alpha Chiller Service Manual",
        "Tenant Alpha emergency repairs must be completed within four hours.\n"
        "For fault code ERR-409B, the chiller reports low water flow. Inspect the "
        "inlet strainer and confirm both isolation valves are open.\n"
        "The manual does not list technician home addresses or personal phone numbers.",
    ),
    "tenant_beta": (
        "tenant_beta_chiller_manual.pdf",
        "Tenant Beta Chiller Service Manual",
        "Tenant Beta emergency repairs must be completed within eight hours.\n"
        "For fault code ERR-409B, the chiller reports low chilled-water flow. "
        "Inspect the return strainer and confirm the circulation pump is running.\n"
        "The manual does not list technician home addresses or personal phone numbers.",
    ),
}


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


def _pdf_bytes(title: str, body: str) -> bytes:
    lines = [title, "", *body.splitlines()]
    commands: list[str] = []
    y = 748
    for line in lines:
        wrapped: list[str] = []
        current = ""
        for word in line.split():
            if current and len(current) + len(word) + 1 > 70:
                wrapped.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        wrapped.append(current)
        for part in wrapped:
            escaped = part.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F1 11 Tf 54 {y} Td ({escaped}) Tj ET")
            y -= 22
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body_bytes in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(body_bytes)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def _write_pdfs() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tenant_id, (filename, title, body) in DOCUMENTS.items():
        path = DATA_DIR / filename
        path.write_bytes(_pdf_bytes(title, body))
        paths[tenant_id] = path
    return paths


@contextmanager
def _tenant_ingest_logins() -> Iterator[dict[str, tuple[str, str]]]:
    settings = get_settings()
    admin_settings = AdminSettings()
    params = conninfo_to_dict(settings.database_url)
    params.update(
        user=admin_settings.postgres_user,
        password=admin_settings.postgres_password.get_secret_value(),
        dbname=admin_settings.postgres_db,
        connect_timeout=10,
    )
    admin = psycopg.connect(make_conninfo(**params), autocommit=True)
    logins: dict[str, tuple[str, str]] = {}
    try:
        with admin.transaction():
            for tenant_id in DOCUMENTS:
                role_suffix = tenant_id.removeprefix("tenant_")
                role_name = f"efm_calibration_{role_suffix}_{secrets.token_hex(5)}"
                password = secrets.token_urlsafe(32)
                admin.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}"
                    ).format(sql.Identifier(role_name), sql.Literal(password))
                )
                logins[tenant_id] = (role_name, password)
                admin.execute(
                    sql.SQL("GRANT efm_app TO {}").format(sql.Identifier(role_name))
                )
                admin.execute("SET ROLE efm_owner")
                admin.execute(
                    "INSERT INTO efm_private.tenant_principals "
                    "(principal_name, tenant_id, allowed_roles) VALUES (%s, %s, %s)",
                    (role_name, tenant_id, ["all", "engineer"]),
                )
                admin.execute("RESET ROLE")
        yield logins
    finally:
        with admin.transaction():
            admin.execute("SET ROLE efm_owner")
            admin.execute(
                "DELETE FROM efm_private.tenant_principals "
                "WHERE principal_name = ANY(%s)",
                ([name for name, _ in logins.values()],),
            )
            admin.execute("RESET ROLE")
        for role_name, _ in logins.values():
            admin.execute(
                sql.SQL("REVOKE efm_app FROM {}").format(sql.Identifier(role_name))
            )
            admin.execute(
                sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
            )
        admin.close()


def _database_url_for(role_name: str, password: str) -> str:
    params = conninfo_to_dict(get_settings().database_url)
    params.update(user=role_name, password=password, connect_timeout=10)
    return make_conninfo(**params)


def main() -> int:
    paths = _write_pdfs()
    old_database_url = os.environ.get("DATABASE_URL")
    try:
        with _tenant_ingest_logins() as logins:
            for tenant_id, path in paths.items():
                role_name, password = logins[tenant_id]
                os.environ["DATABASE_URL"] = _database_url_for(role_name, password)
                get_settings.cache_clear()
                result = ingest(
                    tenant_id=tenant_id,
                    access_role="all",
                    path=path,
                    doc_name=path.name,
                )
                print(
                    f"{tenant_id}: {path} active; version={result.version}, "
                    f"parents={result.parents}, children={result.children}, "
                    f"skipped={result.skipped}"
                )
    finally:
        if old_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_database_url
        get_settings.cache_clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
