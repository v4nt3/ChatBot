"""Ejecuta evaluaciones controladas y crea figuras solo con datos medidos."""

import argparse
import csv
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import chatbot, database, tfidf

CARPETA_RESULTADOS = Path("results")
CARPETA_GRAFICAS = Path("graficas")
ARCHIVO_CASOS = CARPETA_RESULTADOS / "preguntas_prueba.csv"
MODOS = ("tfidf", "llm", "hibrido")
CATEGORIAS = ("Correcta", "Parcial", "Incorrecta", "Sin información")
UMBRALES = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
            0.09, 0.10, 0.12, 0.15, 0.20)
ARCHIVOS = {
    "tiempos_respuesta.csv": ["run_id", "case_id", "owner_id", "document_id", "mode",
                              "elapsed_ms", "seconds", "is_warmup"],
    "metricas_retrieval.csv": ["run_id", "case_id", "document_id", "k", "precision",
                                "recall", "relevant_total", "relevant_retrieved"],
    "evaluacion_umbral.csv": ["run_id", "case_id", "document_id", "threshold", "precision",
                              "recall", "true_positive", "false_positive", "false_negative",
                              "relevant_total"],
    "calidad_respuestas.csv": ["run_id", "case_id", "owner_id", "document_id", "mode",
                               "question", "expected_answer", "answer", "grade"],
    "consumo_tokens.csv": ["run_id", "case_id", "document_id", "mode", "input_tokens",
                           "output_tokens", "total_tokens", "is_warmup"],
}


def preparar_carpetas():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    CARPETA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    for nombre, columnas in ARCHIVOS.items():
        ruta = CARPETA_RESULTADOS / nombre
        if not ruta.exists():
            pd.DataFrame(columns=columnas).to_csv(ruta, index=False)
    if not ARCHIVO_CASOS.exists():
        pd.DataFrame(columns=["case_id", "owner_id", "document_id", "question",
                              "expected_answer", "relevant_chunk_ids"]).to_csv(ARCHIVO_CASOS, index=False)


def agregar_fila(nombre, fila):
    ruta = CARPETA_RESULTADOS / nombre
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        csv.DictWriter(archivo, fieldnames=ARCHIVOS[nombre]).writerow(fila)


def cargar_casos(ruta=ARCHIVO_CASOS):
    casos = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    requeridas = {"case_id", "owner_id", "document_id", "question", "expected_answer",
                  "relevant_chunk_ids"}
    faltantes = requeridas - set(casos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta}: {', '.join(sorted(faltantes))}")
    if len(casos) < 2:
        raise ValueError("Agrega al menos dos preguntas reales para comparar las modalidades.")
    if (casos[["case_id", "owner_id", "document_id", "question", "expected_answer"]] == "").any().any():
        raise ValueError(
            "case_id, owner_id, document_id, question y expected_answer son obligatorios."
        )
    if casos["case_id"].duplicated().any():
        raise ValueError("Cada case_id debe ser único.")
    return casos.to_dict("records")


def ids_relevantes(caso):
    texto = caso["relevant_chunk_ids"].strip()
    if not texto:
        return set()
    try:
        return {int(valor.strip()) for valor in texto.split(";") if valor.strip()}
    except ValueError as error:
        raise ValueError(f"IDs de fragmentos inválidos en el caso {caso['case_id']}.") from error


def calcular_retrieval(run_id, caso, fragmentos):
    relevantes = ids_relevantes(caso)
    ranking = tfidf.buscar_fragmentos(
        fragmentos, caso["question"], cantidad_resultados=max(10, len(fragmentos)),
        similitud_minima=0, document_id=int(caso["document_id"]),
    )
    total_relevantes = len(relevantes)
    for k in range(1, 11):
        primeros = ranking[:k]
        encontrados = sum(item["id"] in relevantes for item in primeros)
        precision = encontrados / len(primeros) if primeros else 0.0
        recall = encontrados / total_relevantes if total_relevantes else ""
        agregar_fila("metricas_retrieval.csv", {
            "run_id": run_id, "case_id": caso["case_id"],
            "document_id": caso["document_id"], "k": k, "precision": precision,
            "recall": recall, "relevant_total": total_relevantes,
            "relevant_retrieved": encontrados,
        })

    puntuaciones = tfidf.buscar_fragmentos(
        fragmentos, caso["question"], cantidad_resultados=len(fragmentos), similitud_minima=0,
        document_id=int(caso["document_id"]),
    )
    for umbral in UMBRALES:
        recuperados = [item for item in puntuaciones if item["score"] > umbral]
        encontrados = sum(item["id"] in relevantes for item in recuperados)
        falsos_positivos = len(recuperados) - encontrados
        falsos_negativos = total_relevantes - encontrados
        precision = encontrados / (encontrados + falsos_positivos) if recuperados else 0.0
        recall = encontrados / total_relevantes if total_relevantes else ""
        agregar_fila("evaluacion_umbral.csv", {
            "run_id": run_id, "case_id": caso["case_id"],
            "document_id": caso["document_id"], "threshold": umbral,
            "precision": precision, "recall": recall, "true_positive": encontrados,
            "false_positive": falsos_positivos, "false_negative": falsos_negativos,
            "relevant_total": total_relevantes,
        })


def preparar_documentos(casos):
    cache = {}
    for caso in casos:
        document_id = int(caso["document_id"])
        owner_id = caso["owner_id"]
        clave = (owner_id, document_id)
        if clave not in cache:
            documento = database.obtener_documento(document_id, owner_id)
            if not documento:
                raise ValueError(
                    f"El documento {document_id} no existe o no pertenece a {owner_id}."
                )
            fragmentos = database.obtener_fragmentos(document_id)
            actuales = {item["id"] for item in fragmentos}
            caso_ids = ids_relevantes(caso)
            if not caso_ids.issubset(actuales):
                raise ValueError(
                    f"El caso {caso['case_id']} incluye IDs de fragmentos ajenos al documento."
                )
            cache[clave] = (documento, fragmentos)
        else:
            actuales = {item["id"] for item in cache[clave][1]}
            if not ids_relevantes(caso).issubset(actuales):
                raise ValueError(
                    f"El caso {caso['case_id']} incluye IDs de fragmentos ajenos al documento."
                )
    return cache


def guardar_tiempo(run_id, caso, modo, resultado, es_calentamiento):
    milisegundos = resultado["elapsed_ms"]
    agregar_fila("tiempos_respuesta.csv", {
        "run_id": run_id, "case_id": caso["case_id"], "owner_id": caso["owner_id"],
        "document_id": caso["document_id"], "mode": modo,
        "elapsed_ms": milisegundos, "seconds": milisegundos / 1000,
        "is_warmup": es_calentamiento,
    })
    if modo in {"llm", "hibrido"}:
        agregar_fila("consumo_tokens.csv", {
            "run_id": run_id, "case_id": caso["case_id"], "document_id": caso["document_id"],
            "mode": modo, "input_tokens": resultado["input_tokens"],
            "output_tokens": resultado["output_tokens"], "total_tokens": resultado["tokens"],
            "is_warmup": es_calentamiento,
        })


def ejecutar(casos):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    documentos = preparar_documentos(casos)
    primero = casos[0]

    # Una consulta de calentamiento por modo, con la misma pregunta y documento;
    # se registra marcada y se excluye de las métricas y gráficas.
    for modo in MODOS:
        resultado = chatbot.consultar(
            primero["owner_id"], int(primero["document_id"]), modo,
            primero["question"], registrar=False,
        )
        guardar_tiempo(run_id, primero, modo, resultado, True)

    for indice, caso in enumerate(casos):
        fragmentos = documentos[(caso["owner_id"], int(caso["document_id"]))][1]
        calcular_retrieval(run_id, caso, fragmentos)
        orden = MODOS[indice % len(MODOS):] + MODOS[:indice % len(MODOS)]
        for modo in orden:
            resultado = chatbot.consultar(
                caso["owner_id"], int(caso["document_id"]), modo, caso["question"]
            )
            guardar_tiempo(run_id, caso, modo, resultado, False)
            agregar_fila("calidad_respuestas.csv", {
                "run_id": run_id, "case_id": caso["case_id"], "owner_id": caso["owner_id"],
                "document_id": caso["document_id"], "mode": modo,
                "question": caso["question"], "expected_answer": caso["expected_answer"],
                "answer": resultado["answer"], "grade": "",
            })
    return run_id


def guardar_figura(figura, nombre):
    figura.tight_layout()
    figura.savefig(CARPETA_GRAFICAS / nombre, dpi=300, bbox_inches="tight")
    plt.close(figura)


def generar_graficas():
    preparar_carpetas()
    creadas = []

    tiempos = pd.read_csv(CARPETA_RESULTADOS / "tiempos_respuesta.csv")
    if not tiempos.empty:
        medidos = tiempos[~tiempos["is_warmup"].astype(str).str.lower().isin(["true", "1"])].copy()
        medidos["seconds"] = pd.to_numeric(medidos["seconds"], errors="coerce")
        datos = [medidos.loc[medidos["mode"] == modo, "seconds"].dropna().values for modo in MODOS]
        presentes = [(modo, valores) for modo, valores in zip(MODOS, datos) if len(valores)]
        if presentes:
            figura, eje = plt.subplots(figsize=(8, 5))
            eje.boxplot([valores for _, valores in presentes], labels=[
                f"{modo.upper()}\n(n={len(valores)})" for modo, valores in presentes
            ], showmeans=True)
            eje.set(title="Distribución del tiempo de respuesta por modalidad",
                    xlabel="Modalidad", ylabel="Tiempo de respuesta (segundos)")
            eje.grid(axis="y", alpha=0.25)
            guardar_figura(figura, "tiempo_respuesta.png")
            creadas.append("tiempo_respuesta.png")

    recuperacion = pd.read_csv(CARPETA_RESULTADOS / "metricas_retrieval.csv")
    if not recuperacion.empty:
        recuperacion["k"] = pd.to_numeric(recuperacion["k"], errors="coerce")
        for columna in ("precision", "recall"):
            recuperacion[columna] = pd.to_numeric(recuperacion[columna], errors="coerce")
        promedio = recuperacion.groupby("k")[["precision", "recall"]].mean()
        if not promedio.empty:
            figura, eje = plt.subplots(figsize=(8, 5))
            eje.plot(promedio.index, promedio["precision"], marker="o", label="Precision@K")
            eje.plot(promedio.index, promedio["recall"], marker="s", label="Recall@K")
            eje.set(title="Precision@K y Recall@K de la recuperación TF-IDF",
                    xlabel="Cantidad de fragmentos recuperados (K)", ylabel="Métrica")
            eje.set_xticks(range(1, 11))
            eje.set_ylim(0, 1)
            eje.grid(alpha=0.25)
            eje.legend()
            guardar_figura(figura, "precision_recall_k.png")
            creadas.append("precision_recall_k.png")

    umbrales = pd.read_csv(CARPETA_RESULTADOS / "evaluacion_umbral.csv")
    if not umbrales.empty:
        umbrales["threshold"] = pd.to_numeric(umbrales["threshold"], errors="coerce")
        for columna in ("precision", "recall"):
            umbrales[columna] = pd.to_numeric(umbrales[columna], errors="coerce")
        promedio = umbrales.groupby("threshold")[["precision", "recall"]].mean()
        if not promedio.empty:
            figura, eje = plt.subplots(figsize=(8, 5))
            eje.plot(promedio.index, promedio["precision"], marker="o", label="Precisión")
            eje.plot(promedio.index, promedio["recall"], marker="s", label="Exhaustividad (Recall)")
            eje.set(title="Precisión y exhaustividad según el umbral TF-IDF",
                    xlabel="Umbral mínimo de similitud", ylabel="Métrica")
            eje.set_ylim(0, 1)
            eje.grid(alpha=0.25)
            eje.legend()
            guardar_figura(figura, "precision_exhaustividad_umbral.png")
            creadas.append("precision_exhaustividad_umbral.png")

    calidad = pd.read_csv(CARPETA_RESULTADOS / "calidad_respuestas.csv", dtype=str,
                          keep_default_na=False)
    if not calidad.empty:
        calidad["grade"] = calidad["grade"].str.strip()
        calidad = calidad[calidad["grade"].isin(CATEGORIAS)]
        if not calidad.empty:
            modos_evaluados = [modo for modo in MODOS if modo in set(calidad["mode"])]
            conteos = pd.crosstab(calidad["mode"], calidad["grade"]).reindex(
                index=modos_evaluados, columns=CATEGORIAS, fill_value=0
            )
            porcentajes = conteos.div(conteos.sum(axis=1).replace(0, pd.NA), axis=0) * 100
            porcentajes = porcentajes.dropna(how="all")
            if not porcentajes.empty:
                figura, eje = plt.subplots(figsize=(9, 5))
                base = pd.Series(0.0, index=porcentajes.index)
                for categoria in CATEGORIAS:
                    valores = porcentajes[categoria].fillna(0)
                    eje.bar(porcentajes.index, valores, bottom=base, label=categoria)
                    base += valores
                etiquetas = [
                    f"{modo}\n(n={int(conteos.loc[modo].sum())})"
                    for modo in porcentajes.index
                ]
                eje.set(title="Clasificación manual de la calidad de las respuestas",
                        xlabel="Modalidad", ylabel="Respuestas evaluadas (%)",
                        xticks=range(len(etiquetas)), xticklabels=etiquetas)
                eje.set_ylim(0, 100)
                eje.legend(title="Evaluación humana", bbox_to_anchor=(1.02, 1), loc="upper left")
                guardar_figura(figura, "calidad_respuestas.png")
                creadas.append("calidad_respuestas.png")

    tokens = pd.read_csv(CARPETA_RESULTADOS / "consumo_tokens.csv")
    if not tokens.empty:
        tokens = tokens[~tokens["is_warmup"].astype(str).str.lower().isin(["true", "1"])].copy()
        tokens["input_tokens"] = pd.to_numeric(tokens["input_tokens"], errors="coerce")
        presentes = [(modo, tokens.loc[tokens["mode"] == modo, "input_tokens"].dropna().values)
                     for modo in ("llm", "hibrido")]
        presentes = [(modo, valores) for modo, valores in presentes if len(valores)]
        if presentes:
            figura, eje = plt.subplots(figsize=(8, 5))
            eje.boxplot([valores for _, valores in presentes], labels=[
                f"{modo.upper()}\n(n={len(valores)})" for modo, valores in presentes
            ], showmeans=True)
            eje.set(title="Tokens de entrada informados por Gemini",
                    xlabel="Modalidad", ylabel="Tokens de entrada por consulta")
            eje.grid(axis="y", alpha=0.25)
            guardar_figura(figura, "consumo_tokens.png")
            creadas.append("consumo_tokens.png")

    return creadas


def exportar_fragmentos(document_id, owner_id):
    if not database.obtener_documento(document_id, owner_id):
        raise ValueError("El documento no existe o no pertenece a ese usuario.")
    ruta = CARPETA_RESULTADOS / f"fragmentos_documento_{document_id}.csv"
    pd.DataFrame(database.obtener_fragmentos(document_id)).to_csv(ruta, index=False)
    return ruta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casos", type=Path, default=ARCHIVO_CASOS,
                        help="CSV con preguntas y juicios de relevancia manuales")
    parser.add_argument("--exportar-fragmentos", type=int,
                        help="Exporta los fragmentos del documento indicado para anotación")
    parser.add_argument("--usuario", help="Propietario del documento para exportar fragmentos")
    parser.add_argument("--solo-graficas", action="store_true",
                        help="Regenera las figuras desde CSV sin ejecutar nuevas consultas")
    argumentos = parser.parse_args()
    preparar_carpetas()
    if argumentos.exportar_fragmentos is not None:
        if not argumentos.usuario:
            parser.error("--exportar-fragmentos requiere --usuario")
        print(exportar_fragmentos(argumentos.exportar_fragmentos, argumentos.usuario))
        return
    if not argumentos.solo_graficas:
        casos = cargar_casos(argumentos.casos)
        run_id = ejecutar(casos)
        print(f"Evaluación completada. run_id={run_id}")
    creadas = generar_graficas()
    if creadas:
        print("Gráficas generadas: " + ", ".join(creadas))
    else:
        print("No hay datos reales suficientes para generar gráficas todavía.")


if __name__ == "__main__":
    main()
