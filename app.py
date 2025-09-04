from flask import Flask, render_template, request, jsonify
import os
from faster_whisper import WhisperModel
import google.generativeai as genai
from gtts import gTTS
from werkzeug.utils import secure_filename
import time
import logging

# ---------------- إعداد Flask ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
AUDIO_RESPONSES_FOLDER = "responses"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_RESPONSES_FOLDER, exist_ok=True)

# ---------------- إعداد Faster-Whisper ----------------
model_name = os.getenv("WHISPER_MODEL", "tiny")
logging.info(f"جاري تحميل موديل Faster-Whisper: {model_name}")
model = WhisperModel(model_name, device="cpu", compute_type="int8")
logging.info("تم تحميل الموديل بنجاح.")

# ---------------- إعداد Gemini ----------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-pro")

# ---------------- المسارات ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400

    try:
        # حفظ الملف الصوتي
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # تحويل الصوت إلى نص
        segments, info = model.transcribe(filepath, language="ar")
        question_text = " ".join([seg.text for seg in segments]).strip()

        # إرسال النص إلى Gemini
        gemini_response = gemini_model.generate_content(question_text)
        answer_text = gemini_response.text.strip()

        # تحويل الرد إلى صوت
        audio_filename = f"response_{int(time.time())}.mp3"
        audio_path = os.path.join(AUDIO_RESPONSES_FOLDER, audio_filename)
        tts = gTTS(text=answer_text, lang="ar", slow=False)
        tts.save(audio_path)

        return jsonify({
            "question": question_text,
            "answer": answer_text,
            "audio_url": f"/responses/{audio_filename}"
        })

    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.route("/responses/<path:filename>")
def get_response_audio(filename):
    return app.send_static_file(os.path.join(AUDIO_RESPONSES_FOLDER, filename))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
