"""Validación, extracción y fragmentación de archivos PDF y TXT."""

import os
import re
import uuid
from pathlib import Path

import fitz
from dotenv import load_dotenv

from . import database, tfidf

load_dotenv()
CARPETA_DOCUMENTOS = Path(os.getenv("DOCUMENTS_DIR", "storage/documents")).resolve()
LIMITE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
TAMANO_FRAGMENTO = 1000
SOLAPAMIENTO = 150


def limpiar_texto(texto):
    texto = texto.replace("\x00", " ")
    return re.sub(r"\s+", " ", texto).strip()


def dividir_fragmentos(texto, pagina=None, tamano=TAMANO_FRAGMENTO, solapamiento=SOLAPAMIENTO):
    if tamano <= 0 or solapamiento < 0 or solapamiento >= tamano:
        raise ValueError("El tamaño y el solapamiento del fragmento no son válidos.")
    texto = limpiar_texto(texto)
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        parte = texto[inicio:fin].strip()
        if parte:
            fragmentos.append({"text": parte, "page": pagina})
        if fin == len(texto):
            break
        inicio = fin - solapamiento
    return fragmentos


def extraer_contenido(ruta, tipo):
    ruta = Path(ruta)
    if tipo == ".txt":
        texto = ruta.read_text(encoding="utf-8-sig")
        return limpiar_texto(texto), dividir_fragmentos(texto)
    if tipo == ".pdf":
        paginas = []
        fragmentos = []
        with fitz.open(ruta) as pdf:
            for numero, pagina in enumerate(pdf, start=1):
                texto = limpiar_texto(pagina.get_text())
                if texto:
                    paginas.append(texto)
                    fragmentos.extend(dividir_fragmentos(texto, pagina=numero))
        return "\n".join(paginas), fragmentos
    raise ValueError("Solo se permiten archivos PDF y TXT.")


def procesar_archivo(owner_id, nombre, contenido):
    extension = Path(nombre).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise ValueError("El archivo debe tener formato PDF o TXT.")
    if not contenido:
        raise ValueError("El archivo está vacío.")
    if len(contenido) > LIMITE_BYTES:
        raise ValueError(f"El archivo supera el límite de {LIMITE_BYTES // (1024 * 1024)} MB.")

    CARPETA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_DOCUMENTOS / f"{uuid.uuid4().hex}{extension}"
    ruta.write_bytes(contenido)
    try:
        texto, fragmentos = extraer_contenido(ruta, extension)
        if not texto or not fragmentos:
            raise ValueError("No fue posible extraer texto del documento.")
        document_id = database.guardar_documento(
            str(owner_id), Path(nombre).name, ruta, extension[1:], texto, fragmentos
        )
        try:
            tfidf.crear_indice(document_id, fragmentos)
        except Exception:
            database.eliminar_documento(document_id, str(owner_id))
            raise
        return {"id": document_id, "name": Path(nombre).name, "chunks": len(fragmentos)}
    except Exception:
        ruta.unlink(missing_ok=True)
        raise
