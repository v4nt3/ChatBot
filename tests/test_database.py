from app import database


def test_persistencia_y_propiedad_de_documento(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "RUTA_BASE_DATOS", tmp_path / "prueba.sqlite3")
    database.inicializar_base_datos()
    document_id = database.guardar_documento(
        "alumno-1", "clase.txt", tmp_path / "clase.txt", "txt", "texto de prueba",
        [{"text": "texto de prueba", "page": None}],
    )
    assert database.obtener_documento(document_id, "alumno-1")["name"] == "clase.txt"
    assert database.obtener_documento(document_id, "alumno-2") is None
    assert database.obtener_fragmentos(document_id)[0]["text"] == "texto de prueba"
    database.guardar_consulta(
        "alumno-1", document_id, "llm", "¿Qué texto?", "Texto de prueba", 250,
        gemini_tokens=12, gemini_input_tokens=8, gemini_output_tokens=4,
    )
    with database.conexion() as base:
        consulta = base.execute(
            "SELECT gemini_input_tokens,gemini_output_tokens FROM conversations"
        ).fetchone()
    assert tuple(consulta) == (8, 4)
