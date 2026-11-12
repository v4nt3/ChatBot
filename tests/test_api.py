from fastapi.testclient import TestClient

from app import database
from app.main import app


def test_health_y_autenticacion_api(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "RUTA_BASE_DATOS", tmp_path / "api.sqlite3")
    monkeypatch.setenv("API_TOKEN", "token-de-prueba")
    monkeypatch.setenv("API_USER_ID", "alumno-api")
    with TestClient(app) as cliente:
        assert cliente.get("/health").json() == {"status": "ok"}
        assert cliente.get("/documents").status_code == 401
        assert cliente.get("/documents", headers={"Authorization": "Bearer incorrecto"}).status_code == 401
        respuesta = cliente.get("/documents", headers={"Authorization": "Bearer token-de-prueba"})
        assert respuesta.status_code == 200
        assert respuesta.json() == []
