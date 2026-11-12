from app import evaluacion


def test_graficas_no_crea_datos_si_los_csv_estan_vacios(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluacion, "CARPETA_RESULTADOS", tmp_path / "results")
    monkeypatch.setattr(evaluacion, "CARPETA_GRAFICAS", tmp_path / "graficas")
    monkeypatch.setattr(evaluacion, "ARCHIVO_CASOS", tmp_path / "results" / "preguntas_prueba.csv")
    evaluacion.preparar_carpetas()
    assert evaluacion.generar_graficas() == []
    assert list((tmp_path / "graficas").glob("*.png")) == []


def test_ids_relevantes_acepta_lista_manual_y_vacio():
    assert evaluacion.ids_relevantes({"case_id": "q1", "relevant_chunk_ids": "2; 5;8"}) == {2, 5, 8}
    assert evaluacion.ids_relevantes({"case_id": "q2", "relevant_chunk_ids": ""}) == set()
