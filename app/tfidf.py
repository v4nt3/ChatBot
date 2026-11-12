"""Búsqueda por TF-IDF y coseno, siguiendo el flujo del taller de clase."""

import os
from pathlib import Path

from dotenv import load_dotenv
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
SIMILITUD_MINIMA = float(os.getenv("SIMILITUD_MINIMA", "0.05"))
CARPETA_INDICES = Path(os.getenv("DOCUMENTS_DIR", "storage/documents")).resolve().parent / "indices"


def crear_indice(document_id, fragmentos):
    """Ajusta el vectorizador una vez y guarda el índice en almacenamiento local."""
    vectorizador = TfidfVectorizer()
    matriz = vectorizador.fit_transform([item["text"] for item in fragmentos])
    CARPETA_INDICES.mkdir(parents=True, exist_ok=True)
    dump((vectorizador, matriz), CARPETA_INDICES / f"{int(document_id)}.joblib")


def eliminar_indice(document_id):
    (CARPETA_INDICES / f"{int(document_id)}.joblib").unlink(missing_ok=True)


def buscar_fragmentos(fragmentos, pregunta, cantidad_resultados=5, similitud_minima=None,
                      document_id=None):
    if not pregunta.strip() or not fragmentos:
        return []
    umbral = SIMILITUD_MINIMA if similitud_minima is None else similitud_minima
    textos = [item["text"] for item in fragmentos]
    ruta_indice = CARPETA_INDICES / f"{int(document_id)}.joblib" if document_id else None
    try:
        if ruta_indice and ruta_indice.is_file():
            vectorizador, matriz = load(ruta_indice)
        else:
            vectorizador = TfidfVectorizer()
            matriz = vectorizador.fit_transform(textos)
    except ValueError:
        return []
    consulta = vectorizador.transform([pregunta])
    puntuaciones = cosine_similarity(consulta, matriz).ravel()
    orden = puntuaciones.argsort()[::-1]
    resultados = []
    for indice in orden:
        if puntuaciones[indice] < umbral:
            break
        resultados.append({**fragmentos[indice], "score": float(puntuaciones[indice])})
        if len(resultados) == cantidad_resultados:
            break
    return resultados
