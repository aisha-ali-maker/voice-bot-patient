# app.py
"""Voice Bot Flask app (PWA ready)
- faster-whisper transcription
- medications.json flexible loader (dict or list)
- optional Gemini (google.generativeai) fallback
- gTTS TTS -> saved mp3 files served from /responses/<name>
- serves /service-worker.js and /manifest.json for PWA
"""
import os
import json
import time
import logging
import re
import difflib
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Optional imports (faster-whisper, genai, gTTS)
try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# try pydub for file conversion fallback (optional; needs ffmpeg installed)
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

# -------------------- app & logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("voice-bot")

app = Flask(__name__, static_folder="static", template_folder="templates")

# -------------------- folders --------------------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
AUDIO_RESPONSES_FOLDER = BASE_DIR / "responses"
for d in (UPLOAD_FOLDER, AUDIO_RESPONSES_FOLDER):
    d.mkdir(parents=True, exist_ok=True)

# -------------------- meds loader (flexible) --------------------
MEDS_FILE = BASE_DIR / "medications.json"

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", re.VERBOSE)

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[آأإٰ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_medications_file(path: Path = MEDS_FILE) -> dict:
    """Load medications.json accepting:
       - dict mapping name -> info
       - list of objects with "name" field
       - list of one-key dicts [{ "اسم": {..} }, ...]
       Returns normalized dict: { name: info_dict_or_string, ... }
    """
    if not path.exists():
        logger.warning(f"{path} not found.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return {}

    meds = {}

    if isinstance(data, dict):
        for name, info in data.items():
            if isinstance(info, str):
                meds[name] = {"description": info}
            elif isinstance(info, dict):
                meds[name] = info
            else:
                meds[name] = {"raw": info}
        logger.info(f"Loaded medications (dict) with {len(meds)} entries.")
        return meds

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # prefer explicit keys: name/اسم/الاسم
                name = item.get("name") or item.get("اسم") or item.get("الاسم")
                if name:
                    entry = {k: v for k, v in item.items() if k not in ("name", "اسم", "الاسم")}
                    meds[name] = entry
                    continue
                # one-key dict: { "أسبرين": { ... } }
                if len(item) == 1:
                    nm, val = next(iter(item.items()))
                    if isinstance(val, dict):
                        meds[nm] = val
                    else:
                        meds[nm] = {"description": val}
                    continue
                logger.warning(f"Skipping unexpected list entry in {path}: {item}")
            else:
                logger.warning(f"Skipping non-dict entry in {path}: {item}")
        logger.info(f"Loaded medications (list) with {len(meds)} entries.")
        return meds

    logger.error(f"Unsupported format for {path}: {type(data)}")
    return {}

def build_med_index_from_dict(meds: dict):
    index = []
    for name, info in meds.items():
        entry = {"key": name, "norms": [], "raw": info}
        entry["norms"].append(normalize_arabic(name))
        if isinstance(info, dict):
            aliases = info.get("aliases") or info.get("alias") or info.get("أسماء_أخرى") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for a in aliases:
                entry["norms"].append(normalize_arabic(a))
        index.append(entry)
    for e in index:
        e["norms"] = list(dict.fromkeys(e["norms"]))
    return index

# load meds & build index
medications_data = load_medications_file()
med_index = build_med_index_from_dict(medications_data)
logger.info(f"Med index built with {len(med_index)} entries.")

# -------------------- faster-whisper model load --------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
whisper_model = None
if WhisperModel is None:
    logger.error("faster-whisper not installed or failed to import. Install faster-whisper.")
else:
    try:
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL}")
        whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        logger.info("Faster-whisper model loaded.")
    except Exception as e:
        logger.exception(f"Failed to load faster-whisper model '{WHISPER_MODEL}': {e}")
        whisper_model = None

# -------------------- Gemini (optional) --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # choose model that is valid in your region (fallback tried previously)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        logger.info("Gemini client configured.")
    except Exception as e:
        logger.exception(f"Failed to configure Gemini: {e}")
        gemini_model = None
else:
    if genai is None:
        logger.warning("google-generativeai library not installed; Gemini unavailable.")
    else:
        logger.warning("GEMINI_API_KEY not set; Gemini disabled.")

# -------------------- helper: transcribe with fallback --------------------
def convert_to_wav_with_pydub(src_path: Path) -> Path:
    if AudioSegment is None:
        raise RuntimeError("pydub not available for conversion")
    audio = AudioSegment.from_file(str(src_path))
    tmp_wav = Path(tempfile.gettempdir()) / f"conv_{int(time.time()*1000)}.wav"
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(str(tmp_wav), format="wav")
    return tmp_wav

def transcribe_file(filepath: Path) -> str:
    """Transcribe using faster-whisper. If failed, try pydub conversion to wav (if available)."""
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded.")
    # try native
    try:
        segments, info = whisper_model.transcribe(str(filepath), language="ar")
        text = " ".join([seg.text for seg in segments]).strip()
        return text
    except Exception as e:
        logger.warning(f"Primary transcription failed: {e}")
        # try convert to wav if possible
        try:
            tmp_wav = convert_to_wav_with_pydub(filepath)
            logger.info(f"Converted audio to WAV at {tmp_wav}")
            segments, info = whisper_model.transcribe(str(tmp_wav), language="ar")
            text = " ".join([seg.text for seg in segments]).strip()
            try:
                tmp_wav.unlink()
            except Exception:
                pass
            return text
        except Exception as e2:
            logger.exception(f"Fallback transcription failed: {e2}")
            raise RuntimeError(f"Transcription failed: {e2}") from e2

# -------------------- med finder --------------------
def find_med_in_text(text: str):
    norm_text = normalize_arabic(text)
    logger.info(f"Normalized question: '{norm_text}'")
    # exact substring
    for entry in med_index:
        for n in entry["norms"]:
            if n and n in norm_text:
                logger.info(f"Exact match found: '{entry['key']}' via norm '{n}'")
                return entry["key"], entry["raw"]
    # token-fuzzy
    tokens = norm_text.split()
    med_names_norm = []
    for entry in med_index:
        med_names_norm.extend(entry["norms"])
    med_names_norm = list(dict.fromkeys(med_names_norm))
    for token in tokens:
        matches = difflib.get_close_matches(token, med_names_norm, n=1, cutoff=0.75)
        if matches:
            matched_norm = matches[0]
            for entry in med_index:
                if matched_norm in entry["norms"]:
                    logger.info(f"Fuzzy token match: token '{token}' -> '{entry['key']}'")
                    return entry["key"], entry["raw"]
    # overall fuzzy
    med_name_list = [e["norms"][0] for e in med_index if e["norms"]]
    overall_matches = difflib.get_close_matches(norm_text, med_name_list, n=1, cutoff=0.6)
    if overall_matches:
        matched_norm = overall_matches[0]
        for entry in med_index:
            if matched_norm == entry["norms"][0]:
                logger.info(f"Overall fuzzy match: '{entry['key']}'")
                return entry["key"], entry["raw"]
    return None, None

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

# serve service-worker and manifest from root for proper scope
@app.route("/service-worker.js")
def service_worker():
    return send_from_directory(app.static_folder, "service-worker.js")

@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json")

@app.route("/upload", methods=["POST"])
def upload_audio():
    # frontend uses form key "file"
    if "file" not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف (key 'file' مفقود)"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "اسم الملف فارغ"}), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        # ensure unique filename with timestamp
        filename = f"{int(time.time()*1000)}_{filename}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        logger.info(f"Saved upload: {filepath}")

        # transcribe
        try:
            question_text = transcribe_file(filepath)
            logger.info(f"Transcribed text: '{question_text}'")
        except Exception as e:
            logger.exception("Transcription error")
            return jsonify({"error": "حدث خطأ أثناء تحويل الصوت إلى نص."}), 500

        if not question_text:
            return jsonify({"error": "النص المستخرج فارغ"}), 400

        # check meds
        med_key, med_info = find_med_in_text(question_text)
        answer_text = ""

        if med_key:
            # structured or string
            if isinstance(med_info, dict):
                dose = med_info.get("الجرعة") or med_info.get("dose") or "غير محددة"
                time_ = med_info.get("الوقت") or med_info.get("time") or "غير محدد"
                notes = med_info.get("ملاحظات") or med_info.get("notes") or ""
                answer_text = f"{med_key}: الجرعة {dose}. الوقت: {time_}. {notes}"
            else:
                answer_text = f"{med_key}: {med_info}"
            logger.info("Answer produced from medications file.")
        else:
            # fallback to Gemini if available
            if gemini_model is not None:
                try:
                    prompt = (
                        "أنت مساعد طبي مقيّد بمعلومات الأدوية الواردة أدناه. "
                        "إذا سأل المستخدم عن مواعيد دواء أو تعليمات تناول دواءٍ من هذه القائمة، "
                        "أجب بناءً على المعلومات فقط وبشكل مختصر وواضح بالعربية.\n\n"
                        f"قائمة الأدوية: {json.dumps(medications_data, ensure_ascii=False)}\n\n"
                        f"سؤال المريض: {question_text}"
                    )
                    gemini_response = gemini_model.generate_content(prompt)
                    answer_text = gemini_response.text.strip() if getattr(gemini_response, "text", None) else ""
                    if not answer_text:
                        answer_text = "⚠️ لم يتم الحصول على رد من Gemini"
                    logger.info("Answer obtained from Gemini.")
                except Exception as e:
                    logger.exception(f"Error calling Gemini: {e}")
                    answer_text = "حدث خطأ عند الاتصال بخدمة Gemini."
            else:
                answer_text = "خدمة Gemini غير مهيّأة حالياً (GEMINI_API_KEY مفقود أو مكتبة غير مثبتة)."

        # TTS -> save mp3
        if gTTS is None:
            logger.warning("gTTS not installed; skipping audio generation.")
            audio_url = ""
        else:
            audio_filename = f"response_{int(time.time()*1000)}.mp3"
            audio_path = AUDIO_RESPONSES_FOLDER / audio_filename
            try:
                tts = gTTS(text=answer_text, lang="ar", slow=False)
                tts.save(str(audio_path))
                audio_url = url_for("get_response_audio", filename=audio_filename)
                logger.info(f"TTS saved to {audio_path}")
            except Exception as e:
                logger.exception(f"TTS error: {e}")
                audio_url = ""

        return jsonify({
            "question": question_text,
            "answer": answer_text,
            "audio_url": audio_url
        })

    finally:
        # cleanup uploaded file
        try:
            if filepath and filepath.exists():
                filepath.unlink()
                logger.info(f"Removed uploaded temp file {filepath}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")

@app.route("/responses/<path:filename>")
def get_response_audio(filename):
    # serve mp3 from responses folder
    return send_from_directory(str(AUDIO_RESPONSES_FOLDER), filename)

# health route
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "whisper_loaded": whisper_model is not None,
        "gemini_configured": gemini_model is not None,
        "meds_loaded": len(med_index)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_flag = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
