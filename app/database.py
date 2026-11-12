"""Persistencia pequeña con SQLite, usando únicamente la biblioteca estándar."""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RUTA_BASE_DATOS = Path(os.getenv("DATABASE_PATH", "storage/docubot.sqlite3"))


@contextmanager
def conexion():
    RUTA_BASE_DATOS.parent.mkdir(parents=True, exist_ok=True)
    base = sqlite3.connect(RUTA_BASE_DATOS)
    base.row_factory = sqlite3.Row
    base.execute("PRAGMA foreign_keys = ON")
    try:
        yield base
        base.commit()
    finally:
        base.close()


def inicializar_base_datos():
    with conexion() as base:
        base.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                full_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                page INTEGER,
                position INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL,
                document_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                retrieved_count INTEGER NOT NULL DEFAULT 0,
                scores TEXT,
                gemini_tokens INTEGER,
                gemini_input_tokens INTEGER,
                gemini_output_tokens INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        columnas = {row["name"] for row in base.execute("PRAGMA table_info(conversations)")}
        if "gemini_input_tokens" not in columnas:
            base.execute("ALTER TABLE conversations ADD COLUMN gemini_input_tokens INTEGER")
        if "gemini_output_tokens" not in columnas:
            base.execute("ALTER TABLE conversations ADD COLUMN gemini_output_tokens INTEGER")
        base.execute("PRAGMA foreign_keys = ON")


def guardar_documento(owner_id, name, file_path, file_type, full_text, chunks):
    with conexion() as base:
        cursor = base.execute(
            "INSERT INTO documents(owner_id,name,file_path,file_type,full_text) VALUES(?,?,?,?,?)",
            (owner_id, name, str(file_path), file_type, full_text),
        )
        document_id = cursor.lastrowid
        base.executemany(
            "INSERT INTO chunks(document_id,text,page,position) VALUES(?,?,?,?)",
            [(document_id, item["text"], item.get("page"), i) for i, item in enumerate(chunks)],
        )
        return document_id


def listar_documentos(owner_id):
    with conexion() as base:
        return [dict(row) for row in base.execute(
            "SELECT id,name,file_type,created_at FROM documents WHERE owner_id=? ORDER BY id DESC",
            (owner_id,),
        )]


def obtener_documento(document_id, owner_id):
    with conexion() as base:
        row = base.execute(
            "SELECT * FROM documents WHERE id=? AND owner_id=?", (document_id, owner_id)
        ).fetchone()
        return dict(row) if row else None


def obtener_fragmentos(document_id):
    with conexion() as base:
        return [dict(row) for row in base.execute(
            "SELECT id,text,page,position FROM chunks WHERE document_id=? ORDER BY position",
            (document_id,),
        )]


def eliminar_documento(document_id, owner_id):
    with conexion() as base:
        base.execute("PRAGMA foreign_keys = ON")
        row = base.execute(
            "SELECT file_path FROM documents WHERE id=? AND owner_id=?", (document_id, owner_id)
        ).fetchone()
        if not row:
            return None
        base.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        base.execute("DELETE FROM documents WHERE id=?", (document_id,))
        return row["file_path"]


def guardar_consulta(owner_id, document_id, mode, question, answer, elapsed_ms,
                     retrieved_count=0, scores=None, gemini_tokens=None,
                     gemini_input_tokens=None, gemini_output_tokens=None):
    import json

    with conexion() as base:
        base.execute(
            """INSERT INTO conversations
            (owner_id,document_id,mode,question,answer,elapsed_ms,retrieved_count,scores,
             gemini_tokens,gemini_input_tokens,gemini_output_tokens)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (owner_id, document_id, mode, question, answer, elapsed_ms, retrieved_count,
             json.dumps(scores or [], ensure_ascii=False), gemini_tokens,
             gemini_input_tokens, gemini_output_tokens),
        )


def resumen_estadisticas(owner_id):
    with conexion() as base:
        docs = base.execute("SELECT COUNT(*) FROM documents WHERE owner_id=?", (owner_id,)).fetchone()[0]
        consultas = base.execute(
            "SELECT COUNT(*) FROM conversations WHERE owner_id=?", (owner_id,)
        ).fetchone()[0]
        modos = [dict(row) for row in base.execute(
            "SELECT mode,COUNT(*) AS total FROM conversations WHERE owner_id=? GROUP BY mode",
            (owner_id,),
        )]
        return {"documents": docs, "queries": consultas, "modes": modos}
