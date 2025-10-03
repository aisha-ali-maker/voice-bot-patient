import os
import time
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from faster_whisper import WhisperModel
import google.generativeai as genai
from gtts import gTTS
from werkzeug.utils import secure_filename

# ---------------- إعدادات ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-bot")

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESPONSES_FOLDER = "responses"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESPONSES_FOLDER, exist_ok=True)

# ---------------- Whisper ----------------
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ---------------- Gemini ----------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ الموديل الصحيح (جرّب flash أو pro)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "لا يوجد ملف مرفوع"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    logger.info(f"Saved upload: {filepath}")

    # ---------------- تفريغ الكلام (STT) ----------------
    segments, info = whisper_model.transcribe(filepath, beam_size=5)
    transcript = " ".join([seg.text for seg in segments])
    logger.info(f"Transcript: {transcript}")

    # ---------------- Gemini يجاوب ----------------
    try:
        response = gemini_model.generate_content(transcript)
        bot_answer = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        bot_answer = "حدث خطأ في توليد الرد من Gemini."

    # ---------------- تحويل النص إلى كلام (TTS) ----------------
    timestamp = str(int(time.time()))
    audio_filename = f"response_{timestamp}.mp3"
    audio_path = os.path.join(RESPONSES_FOLDER, audio_filename)

    tts = gTTS(bot_answer, lang="ar")
    tts.save(audio_path)
    logger.info(f"TTS saved to {audio_path}")

    # ---------------- حذف الملف المرفوع ----------------
    try:
        os.remove(filepath)
        logger.info(f"Removed temp file {filepath}")
    except Exception as e:
        logger.warning(f"Could not remove temp file {filepath}: {e}")

    return jsonify({
        "question": transcript,
        "answer": bot_answer,
        "audio_url": f"/responses/{audio_filename}"
    })

@app.route("/responses/<filename>")
def serve_audio(filename):
    return send_from_directory(RESPONSES_FOLDER, filename)

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
