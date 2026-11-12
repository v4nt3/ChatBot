from app.documentos import dividir_fragmentos, extraer_contenido


def test_lee_txt_y_divide_en_fragmentos(tmp_path):
    archivo = tmp_path / "pequeno.txt"
    archivo.write_text("árbol " * 300, encoding="utf-8")
    texto, fragmentos = extraer_contenido(archivo, ".txt")
    assert "árbol" in texto
    assert len(fragmentos) > 1
    assert all(len(fragmento["text"]) <= 1000 for fragmento in fragmentos)


def test_extrae_texto_pdf_con_pagina(tmp_path):
    import fitz

    archivo = tmp_path / "pequeno.pdf"
    documento = fitz.open()
    pagina = documento.new_page()
    pagina.insert_text((72, 72), "Contenido de prueba en la página uno")
    documento.save(archivo)
    documento.close()
    texto, fragmentos = extraer_contenido(archivo, ".pdf")
    assert "Contenido de prueba" in texto
    assert fragmentos[0]["page"] == 1


def test_solapamiento_conserva_contexto():
    fragmentos = dividir_fragmentos("a" * 2200, tamano=1000, solapamiento=150)
    assert [len(item["text"]) for item in fragmentos] == [1000, 1000, 500]
