from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
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

# ---------------- تحميل بيانات الأدوية ----------------
MEDS_FILE = "medications.json"
if os.path.exists(MEDS_FILE):
    with open(MEDS_FILE, "r", encoding="utf-8") as f:
        medications_data = json.load(f)
else:
    medications_data = {}

# ---------------- إعداد Faster-Whisper ----------------
model_name = os.getenv("WHISPER_MODEL", "tiny")
logging.info(f"جاري تحميل موديل Faster-Whisper: {model_name}")
model = WhisperModel(model_name, device="cpu", compute_type="int8")
logging.info("تم تحميل الموديل بنجاح.")

# ---------------- إعداد Gemini ----------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ لم يتم العثور على GEMINI_API_KEY في Environment variables")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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

        if not question_text:
            return jsonify({"error": "النص المستخرج فارغ"}), 400

        # ---------------- البحث في ملف الأدوية ----------------
        answer_text = ""
        for med_name, med_info in medications_data.items():
            if med_name in question_text:
                الجرعة = med_info.get("الجرعة", "غير محددة")
                الوقت = med_info.get("الوقت", "غير محدد")
                ملاحظات = med_info.get("ملاحظات", "لا توجد ملاحظات")
                answer_text = f"{med_name}: الجرعة {جرعة}, الوقت {الوقت}, الملاحظات: {ملاحظات}"
                break

        # ---------------- Gemini لو ما لقاهاش ----------------
        if not answer_text:
            gemini_response = gemini_model.generate_content(question_text)
            answer_text = gemini_response.text.strip() if gemini_response.text else "⚠️ لم يتم الحصول على رد من Gemini"

        # ---------------- تحويل الرد إلى صوت ----------------
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
    return send_from_directory(AUDIO_RESPONSES_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
