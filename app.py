import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from faster_whisper import WhisperModel

# ---------------------------
# إعدادات عامة
# ---------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-bot")

# تحميل ملف الأدوية
import json
try:
    with open("medications.json", "r", encoding="utf-8") as f:
        medications = json.load(f)
    med_names = [m["name"] for m in medications]
    logger.info(f"Loaded medications: {med_names}")
except Exception as e:
    logger.error(f"Error loading medications: {e}")
    medications = []

# تحميل نموذج Faster Whisper
logger.info("Loading faster-whisper model: tiny")
whisper_model = WhisperModel("tiny")
logger.info("Faster-whisper model loaded.")

# ---------------------------
# المسارات الأساسية
# ---------------------------

# الصفحة الرئيسية
@app.route("/")
def index():
    return render_template("index.html")

# manifest.json
@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")

# service-worker.js
@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

# ---------------------------
# API: رفع صوت → نص
# ---------------------------
@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    path = "temp_audio.wav"
    audio_file.save(path)

    segments, _ = whisper_model.transcribe(path)
    text = " ".join([seg.text for seg in segments])

    logger.info(f"User said: {text}")
    return jsonify({"text": text})

# ---------------------------
# تشغيل التطبيق
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
