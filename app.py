# app.py (محدّث) 
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from faster_whisper import WhisperModel
import google.generativeai as genai
from gtts import gTTS
from werkzeug.utils import secure_filename
import time
import logging
import re
import difflib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-bot")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
AUDIO_RESPONSES_FOLDER = "responses"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_RESPONSES_FOLDER, exist_ok=True)

# ---------------- تحميل بيانات الأدوية ----------------
MEDS_FILE = "medications.json"

def load_medications():
    if os.path.exists(MEDS_FILE):
        try:
            with open(MEDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded medications: {list(data.keys())}")
            return data
        except Exception as e:
            logger.error(f"Error loading medications.json: {e}")
            return {}
    else:
        logger.warning("medications.json not found.")
        return {}

medications_data = load_medications()


# ---------------- دوال تطبيع النص العربي ----------------
ARABIC_DIACRITICS = re.compile("""
                             [\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]
                             """, re.VERBOSE)

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    # remove tashkeel/diacritics
    text = ARABIC_DIACRITICS.sub("", text)
    # normalize alef variations
    text = re.sub(r"[آأإٰ]", "ا", text)
    # normalize taa marbuta to ه? (usually keep as ة) — keep as is
    # normalize ya/aa variations
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    # remove punctuation and extra symbols
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- بناء فهرس للأسماء (normalized) ----------------
def build_med_index(meds: dict):
    """
    Returns list of entries:
    [
      {
        "key": original_name,
        "norms": [normalized_name, ...aliases_normalized],
        "raw": med_info (dict or string)
      }, ...
    ]
    """
    index = []
    for name, info in meds.items():
        entry = {"key": name, "norms": [], "raw": info}
        entry["norms"].append(normalize_arabic(name))
        # if info is dict and has aliases field, include them
        if isinstance(info, dict):
            aliases = info.get("aliases") or info.get("alias") or info.get("أسماء_أخرى") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for a in aliases:
                entry["norms"].append(normalize_arabic(a))
        index.append(entry)
    # deduplicate norms
    for e in index:
        e["norms"] = list(dict.fromkeys(e["norms"]))
    return index

med_index = build_med_index(medications_data)
logger.info(f"Med index built with {len(med_index)} entries.")


# ---------------- إعداد Faster-Whisper ----------------
model_name = os.getenv("WHISPER_MODEL", "tiny")
logger.info(f"Loading faster-whisper model: {model_name}")
model = WhisperModel(model_name, device="cpu", compute_type="int8")
logger.info("Faster-whisper model loaded.")


# ---------------- إعداد Gemini ----------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.warning("GEMINI_API_KEY not found in environment. Set it on Render Secrets.")
else:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")



# ---------------- دالة للبحث عن دواء في النص ----------------
def find_med_in_text(text: str):
    """
    Returns tuple (found_med_key, med_info) or (None, None)
    Strategy:
      1) normalize text
      2) exact substring check for each normalized med name / aliases
      3) token-by-token close match (difflib) against med norms
    """
    norm_text = normalize_arabic(text)
    logger.info(f"Normalized question: '{norm_text}'")

    # 1) exact substring match
    for entry in med_index:
        for n in entry["norms"]:
            if n and n in norm_text:
                logger.info(f"Exact match found: '{entry['key']}' via norm '{n}'")
                return entry["key"], entry["raw"]

    # 2) token-based matching (words)
    tokens = norm_text.split()
    med_names_norm = []
    for entry in med_index:
        med_names_norm.extend(entry["norms"])
    med_names_norm = list(dict.fromkeys(med_names_norm))  # unique

    for token in tokens:
        # try to find close match for token among med names (cutoff 0.75)
        matches = difflib.get_close_matches(token, med_names_norm, n=1, cutoff=0.75)
        if matches:
            matched_norm = matches[0]
            # find entry with this norm
            for entry in med_index:
                if matched_norm in entry["norms"]:
                    logger.info(f"Fuzzy token match: token '{token}' -> '{entry['key']}' (norm '{matched_norm}')")
                    return entry["key"], entry["raw"]

    # 3) overall fuzzy match: check whole med names vs whole text
    # take med original names normalized
    med_name_list = [e["norms"][0] for e in med_index if e["norms"]]
    overall_matches = difflib.get_close_matches(norm_text, med_name_list, n=1, cutoff=0.6)
    if overall_matches:
        matched_norm = overall_matches[0]
        for entry in med_index:
            if matched_norm == entry["norms"][0]:
                logger.info(f"Overall fuzzy match: '{entry['key']}'")
                return entry["key"], entry["raw"]

    return None, None


# ---------------- المسارات ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف بصمة 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Saved upload: {filepath}")

        # transcribe
        segments, info = model.transcribe(filepath, language="ar")
        question_text = " ".join([seg.text for seg in segments]).strip()
        logger.info(f"Transcribed text: '{question_text}'")

        if not question_text:
            return jsonify({"error": "النص المستخرج فارغ"}), 400

        # search meds first
        med_key, med_info = find_med_in_text(question_text)
        answer_text = ""

        if med_key:
            # if med_info is dict, build structured answer; if string, return it directly
            if isinstance(med_info, dict):
                dose = med_info.get("الجرعة") or med_info.get("dose") or "غير محددة"
                time_ = med_info.get("الوقت") or med_info.get("time") or "غير محدد"
                notes = med_info.get("ملاحظات") or med_info.get("notes") or ""
                answer_text = f"{med_key}: الجرعة {dose}. الوقت: {time_}. {notes}"
            else:
                # med_info might be a simple descriptive string
                answer_text = f"{med_key}: {med_info}"
            logger.info(f"Answer from meds file: {answer_text}")
        else:
            # fallback to Gemini (if configured)
            if api_key:
                prompt = (
                    "أنت مساعد طبي محدد بمعلومات الدواء الموجودة في القائمة التالية. "
                    "إذا سأل المستخدم عن أحد الأدوية في هذه القائمة، جاوب حسب هذه المعلومات فقط. "
                    "أجب بالعربية وباختصار.\n\n"
                    f"قائمة الأدوية: {json.dumps(medications_data, ensure_ascii=False)}\n\n"
                    f"سؤال المريض: {question_text}"
                )
                try:
                    gemini_response = gemini_model.generate_content(prompt)
                    answer_text = gemini_response.text.strip() if gemini_response.text else "⚠️ لم يتم الحصول على رد من Gemini"
                    logger.info("Answer from Gemini obtained.")
                except Exception as e:
                    logger.error(f"Error calling Gemini: {e}")
                    answer_text = "حدث خطأ عند الاتصال بخدمة Gemini."
            else:
                answer_text = "خدمة Gemini غير مهيّأة حالياً (GEMINI_API_KEY مفقود)."

        # TTS
        audio_filename = f"response_{int(time.time())}.mp3"
        audio_path = os.path.join(AUDIO_RESPONSES_FOLDER, audio_filename)
        tts = gTTS(text=answer_text, lang="ar", slow=False)
        tts.save(audio_path)
        logger.info(f"TTS saved to {audio_path}")

        return jsonify({
            "question": question_text,
            "answer": answer_text,
            "audio_url": f"/responses/{audio_filename}"
        })

    except Exception as e:
        logger.exception("Processing error")
        return jsonify({"error": "حدث خطأ داخلي في الخادم"}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Removed temp file {filepath}")


@app.route("/responses/<path:filename>")
def get_response_audio(filename):
    return send_from_directory(AUDIO_RESPONSES_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
