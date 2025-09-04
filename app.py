from flask import Flask, render_template, request, redirect, url_for
import os
import whisper
import google.generativeai as genai
from gtts import gTTS
from werkzeug.utils import secure_filename
import time

# إعداد Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# اختيار موديل Whisper من متغير بيئة أو "tiny" افتراضياً
model_name = os.getenv("WHISPER_MODEL", "tiny")
print(f"Loading Whisper model: {model_name}")
model = whisper.load_model(model_name)

# إعداد Gemini (باستخدام Environment Variable)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-pro")

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return redirect(url_for("index"))

    file = request.files["audio"]
    if file.filename == "":
        return redirect(url_for("index"))

    # حفظ الملف الصوتي المرفوع
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # تحويل الصوت إلى نص عبر Whisper
    result = model.transcribe(filepath, language="ar")
    text = result["text"]

    # إرسال النص إلى Gemini
    response = gemini.generate_content(text)
    reply = response.text

    # تحويل الرد إلى صوت
    audio_reply = os.path.join(app.config["UPLOAD_FOLDER"], f"reply_{int(time.time())}.mp3")
    tts = gTTS(reply, lang="ar")
    tts.save(audio_reply)

    # إرجاع الرد والملف الصوتي
    return render_template("index.html", text=text, reply=reply, audio=audio_reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
