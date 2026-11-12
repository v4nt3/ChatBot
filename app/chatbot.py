"""Coordina TF-IDF, Gemini y las consultas guardadas en SQLite."""

import os
import time

from dotenv import load_dotenv

from . import database, gemini, tfidf

load_dotenv()
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "100000"))


def consultar(owner_id, document_id, modo, pregunta, registrar=True):
    documento = database.obtener_documento(document_id, str(owner_id))
    if not documento:
        raise LookupError("No se encontró ese documento para este usuario.")
    if modo not in {"tfidf", "llm", "hibrido"}:
        raise ValueError("El modo debe ser tfidf, llm o hibrido.")
    inicio = time.perf_counter()
    fuentes = []
    tokens_entrada = None
    tokens_salida = None
    tokens_total = None
    if modo == "tfidf":
        fuentes = tfidf.buscar_fragmentos(
            database.obtener_fragmentos(document_id), pregunta, document_id=document_id
        )
        if fuentes:
            respuesta = "\n\n".join(
                f"[{documento['name']} | página {f['page'] or 'TXT'} | similitud {f['score']:.3f}]\n{f['text']}"
                for f in fuentes
            )
        else:
            respuesta = "No encontré información relevante en el documento."
    elif modo == "llm":
        if len(documento["full_text"]) > MAX_CONTEXT_CHARS:
            raise ValueError(
                f"El documento supera el límite local de contexto ({MAX_CONTEXT_CHARS} caracteres). "
                "No se envió a Gemini ni se analizó parcialmente."
            )
        respuesta, tokens_entrada, tokens_salida, tokens_total = gemini.generar_respuesta(
            pregunta, documento["full_text"]
        )
        fuentes = [{"document": documento["name"], "page": "documento completo"}]
    else:
        fuentes = tfidf.buscar_fragmentos(
            database.obtener_fragmentos(document_id), pregunta, document_id=document_id
        )
        if not fuentes:
            respuesta = "No encontré información relevante en el documento."
        else:
            contexto = "\n\n".join(f"Página {f['page'] or 'TXT'}: {f['text']}" for f in fuentes)
            respuesta, tokens_entrada, tokens_salida, tokens_total = gemini.generar_respuesta(
                pregunta, contexto
            )
    elapsed_ms = round((time.perf_counter() - inicio) * 1000)
    if registrar:
        database.guardar_consulta(
            str(owner_id), document_id, modo, pregunta, respuesta, elapsed_ms,
            sum("score" in f for f in fuentes),
            [f["score"] for f in fuentes if "score" in f], tokens_total,
            tokens_entrada, tokens_salida,
        )
    return {
        "answer": respuesta,
        "sources": fuentes,
        "elapsed_ms": elapsed_ms,
        "tokens": tokens_total,
        "input_tokens": tokens_entrada,
        "output_tokens": tokens_salida,
    }
