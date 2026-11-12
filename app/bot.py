"""Interfaz de Telegram; llama las funciones compartidas con la API."""

import os

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)

from . import chatbot, database, documentos

load_dotenv()
selecciones = {}


def usuario_id(update):
    return str(update.effective_user.id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hola, soy DocuBot AI. Envíame un PDF o TXT y luego usa /documentos y /modo. "
        "Comandos: /documentos, /modo, /estadisticas."
    )


async def recibir_documento(update: Update, context: ContextTypes.DEFAULT_TYPE):
    archivo = update.message.document
    if archivo.file_size and archivo.file_size > documentos.LIMITE_BYTES:
        await update.message.reply_text("El documento supera el límite permitido de 10 MB.")
        return
    try:
        archivo_telegram = await context.bot.get_file(archivo.file_id)
        contenido = bytes(await archivo_telegram.download_as_bytearray())
        resultado = documentos.procesar_archivo(usuario_id(update), archivo.file_name or "", contenido)
        await update.message.reply_text(
            f"Documento procesado: {resultado['name']} ({resultado['chunks']} fragmentos)."
        )
    except Exception as error:
        await update.message.reply_text(f"No pude procesar el documento: {error}")


async def documentos_usuario(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lista = database.listar_documentos(usuario_id(update))
    if not lista:
        await update.message.reply_text("Todavía no has cargado documentos.")
        return
    teclado = [[InlineKeyboardButton(d["name"], callback_data=f"doc:{d['id']}")] for d in lista]
    await update.message.reply_text("Selecciona un documento:", reply_markup=InlineKeyboardMarkup(teclado))


async def modos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    teclado = [[InlineKeyboardButton(nombre, callback_data=f"modo:{modo}")]
               for modo, nombre in (("tfidf", "TF-IDF"), ("llm", "LLM"), ("hibrido", "Híbrido"))]
    await update.message.reply_text("Selecciona el modo:", reply_markup=InlineKeyboardMarkup(teclado))


async def seleccionar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    usuario = str(query.from_user.id)
    tipo, valor = query.data.split(":", 1)
    actual = selecciones.setdefault(usuario, {"mode": None, "document_id": None})
    if tipo == "doc":
        documento = database.obtener_documento(int(valor), usuario)
        if not documento:
            await query.edit_message_text("Ese documento no está disponible para tu usuario.")
            return
        actual["document_id"] = int(valor)
        mensaje = f"Documento seleccionado: {documento['name']}"
    else:
        actual["mode"] = valor
        mensaje = f"Modo seleccionado: {valor.upper()}"
    await query.edit_message_text(mensaje)


async def estadisticas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    datos = database.resumen_estadisticas(usuario_id(update))
    await update.message.reply_text(
        f"Documentos: {datos['documents']}\nConsultas: {datos['queries']}\nModos: {datos['modes']}"
    )


async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    actual = selecciones.get(usuario_id(update), {})
    if not actual.get("document_id"):
        await update.message.reply_text("Primero selecciona un documento con /documentos.")
        return
    if not actual.get("mode"):
        await update.message.reply_text("Primero selecciona el modo de consulta con /modo.")
        return
    try:
        resultado = chatbot.consultar(
            usuario_id(update), actual["document_id"], actual["mode"], update.message.text
        )
        respuesta = resultado["answer"]
        if resultado["sources"] and actual["mode"] != "tfidf":
            origenes = sorted({
                f"{fuente.get('document', '')} página {fuente.get('page') or 'TXT'}"
                for fuente in resultado["sources"]
            })
            respuesta += "\n\nFuentes: " + "; ".join(origenes)
        for inicio in range(0, len(respuesta), 4000):
            await update.message.reply_text(respuesta[inicio:inicio + 4000])
    except Exception as error:
        await update.message.reply_text(f"No pude completar la consulta: {error}")


def main():
    database.inicializar_base_datos()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Configura TELEGRAM_BOT_TOKEN en .env.")
    tiempo_espera = float(os.getenv("TELEGRAM_READ_TIMEOUT", "30"))
    tiempo_consulta = float(os.getenv("TELEGRAM_GET_UPDATES_READ_TIMEOUT", "40"))
    aplicacion = (
        Application.builder()
        .token(token)
        .connect_timeout(tiempo_espera)
        .read_timeout(tiempo_espera)
        .write_timeout(tiempo_espera)
        .pool_timeout(tiempo_espera)
        .media_write_timeout(max(tiempo_espera, 60))
        .get_updates_connect_timeout(tiempo_espera)
        .get_updates_read_timeout(tiempo_consulta)
        .get_updates_write_timeout(tiempo_espera)
        .get_updates_pool_timeout(tiempo_espera)
        .build()
    )
    aplicacion.add_handler(CommandHandler("start", start))
    aplicacion.add_handler(CommandHandler("documentos", documentos_usuario))
    aplicacion.add_handler(CommandHandler("modo", modos))
    aplicacion.add_handler(CommandHandler("estadisticas", estadisticas))
    aplicacion.add_handler(CallbackQueryHandler(seleccionar))
    aplicacion.add_handler(MessageHandler(filters.Document.ALL, recibir_documento))
    aplicacion.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    aplicacion.run_polling()


if __name__ == "__main__":
    main()
