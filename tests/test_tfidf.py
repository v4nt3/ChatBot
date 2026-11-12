from app.tfidf import buscar_fragmentos


def test_recupera_fragmento_relevante_y_ordenado():
    fragmentos = [
        {"text": "Los gatos son mamíferos domésticos.", "page": 1},
        {"text": "La fotosíntesis transforma luz solar en energía.", "page": 2},
    ]
    resultado = buscar_fragmentos(fragmentos, "¿Qué es la fotosíntesis?", similitud_minima=0)
    assert resultado[0]["page"] == 2
    assert resultado[0]["score"] > resultado[1]["score"]


def test_umbral_descarta_resultados_sin_relacion():
    fragmentos = [{"text": "El océano tiene agua salada.", "page": 1}]
    assert buscar_fragmentos(fragmentos, "fútbol y baloncesto", similitud_minima=0.1) == []
