"""API REST de DocuBot AI."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import chatbot, database, documentos, tfidf

load_dotenv()


@asynccontextmanager
async def ciclo_vida(app):
    database.inicializar_base_datos()
    yield


app = FastAPI(
    title="DocuBot AI",
    description="Consulta académica de documentos PDF y TXT",
    lifespan=ciclo_vida,
)


def usuario_api(authorization: str | None = Header(default=None)):
    token = os.getenv("API_TOKEN", "")
    user_id = os.getenv("API_USER_ID", "")
    if not token or not user_id or not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Autenticación requerida.")
    if not hmac.compare_digest(authorization[7:], token):
        raise HTTPException(401, "Token inválido.")
    return user_id


class Consulta(BaseModel):
    document_id: int
    mode: str = Field(pattern="^(tfidf|llm|hibrido)$")
    question: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), owner_id=Depends(usuario_api)):
    try:
        contenido = await file.read(documentos.LIMITE_BYTES + 1)
        return documentos.procesar_archivo(owner_id, file.filename or "", contenido)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(422, f"No se pudo procesar el archivo: {error}") from error


@app.get("/documents")
def list_documents(owner_id=Depends(usuario_api)):
    return database.listar_documentos(owner_id)


@app.get("/documents/{document_id}")
def get_document(document_id: int, owner_id=Depends(usuario_api)):
    documento = database.obtener_documento(document_id, owner_id)
    if not documento:
        raise HTTPException(404, "Documento no encontrado.")
    documento.pop("full_text", None)
    documento.pop("file_path", None)
    return documento


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, owner_id=Depends(usuario_api)):
    ruta = database.eliminar_documento(document_id, owner_id)
    if not ruta:
        raise HTTPException(404, "Documento no encontrado.")
    ruta = Path(ruta).resolve()
    if ruta.is_relative_to(documentos.CARPETA_DOCUMENTOS) and ruta.is_file():
        ruta.unlink()
    tfidf.eliminar_indice(document_id)
    return {"deleted": True}


@app.post("/chat/query")
def chat_query(solicitud: Consulta, owner_id=Depends(usuario_api)):
    try:
        return chatbot.consultar(owner_id, solicitud.document_id, solicitud.mode, solicitud.question)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/analytics/summary")
def analytics(owner_id=Depends(usuario_api)):
    return database.resumen_estadisticas(owner_id)
