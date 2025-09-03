from flask import Flask, render_template, request, redirect, url_for
import os
import whisper
import google.generativeai as genai
from gtts import gTTS
from werkzeug.utils import secure_filename

# إعداد Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# تحميل نموذج Whisper
model = whisper.load_model("base")

# إعداد Gemini
genai.configure(api_key="ضع_مفتاح_API_هنا")  # ← غيرها بمفتاحك
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
    tts = gTTS(reply, lang="ar")
    audio_reply = os.path.join(app.config["UPLOAD_FOLDER"], "reply.mp3")
    tts.save(audio_reply)

    return render_template("index.html", text=text, reply=reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
