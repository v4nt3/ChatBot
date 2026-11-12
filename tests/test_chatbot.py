import pytest

from app import chatbot


@pytest.fixture
def datos_documento(monkeypatch):
    documento = {"id": 4, "name": "clase.txt", "full_text": "La energía solar usa luz."}
    fragmentos = [{"text": "La energía solar usa luz.", "page": None, "position": 0}]
    monkeypatch.setattr(chatbot.database, "obtener_documento",
                        lambda document_id, owner: documento if owner == "alumno" else None)
    monkeypatch.setattr(chatbot.database, "obtener_fragmentos", lambda document_id: fragmentos)
    monkeypatch.setattr(chatbot.database, "guardar_consulta", lambda *args, **kwargs: None)
    monkeypatch.setattr(chatbot.tfidf, "buscar_fragmentos", lambda *args, **kwargs: [
        {**fragmentos[0], "score": 0.8}
    ])
    return documento


def test_modo_tfidf_no_llama_gemini(monkeypatch, datos_documento):
    def error_si_llama(*args):
        raise AssertionError("TF-IDF no debe llamar Gemini")

    monkeypatch.setattr(chatbot.gemini, "generar_respuesta", error_si_llama)
    resultado = chatbot.consultar("alumno", 4, "tfidf", "¿Qué energía?")
    assert resultado["sources"][0]["score"] == 0.8


def test_modo_llm_usa_documento_completo(monkeypatch, datos_documento):
    monkeypatch.setattr(chatbot.gemini, "generar_respuesta",
                        lambda pregunta, contexto: ("Respuesta", 8, 4, 12))
    resultado = chatbot.consultar("alumno", 4, "llm", "¿Qué energía?")
    assert resultado["answer"] == "Respuesta"
    assert resultado["tokens"] == 12
    assert resultado["input_tokens"] == 8
    assert resultado["sources"][0]["page"] == "documento completo"


def test_modo_hibrido_envia_fragmentos_a_gemini(monkeypatch, datos_documento):
    contextos = []
    monkeypatch.setattr(chatbot.gemini, "generar_respuesta",
                        lambda pregunta, contexto: (contextos.append(contexto) or "Respuesta", None, None, None))
    chatbot.consultar("alumno", 4, "hibrido", "¿Qué energía?")
    assert "La energía solar" in contextos[0]


def test_documento_inexistente_o_de_otro_usuario_es_rechazado(datos_documento):
    with pytest.raises(LookupError):
        chatbot.consultar("otra-persona", 4, "tfidf", "pregunta")
