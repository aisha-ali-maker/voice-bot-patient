import os
import whisper
import sounddevice as sd
import wavio
from gtts import gTTS

# تحميل نموذج Whisper الخفيف
model = whisper.load_model("base")

# قاعدة بيانات الأسئلة والأجوبة
qa_data = {
    "السلام عليكم": "وعليكم السلام ورحمة الله وبركاته",
    "شنو اسمك": "انا مساعدك الصوتي",
    "كيف حالك": "الحمد لله، بخير. شكراً لسؤالك",
    "اعطني دواء للصداع": "يمكنك تناول الباراسيتامول بجرعة مناسبة حسب حالتك، ولكن استشير الطبيب أولاً"
}

# تسجيل الصوت
def record_audio(filename, duration=5, fs=16000):
    print("🎙️ ابدأ الكلام الآن...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)
    print("✅ انتهى التسجيل")

# تحويل الصوت إلى نص
def speech_to_text(filename):
    result = model.transcribe(filename, language="ar")
    return result["text"].strip()

# البحث عن الجواب
def get_answer(question):
    for q, a in qa_data.items():
        if q in question:
            return a
    return None

# تحويل النص إلى صوت باستخدام gTTS وتشغيله مباشرة على Windows
def text_to_speech(text):
    temp_path = os.path.join(os.getcwd(), "temp_audio.mp3")
    tts = gTTS(text=text, lang="ar")
    tts.save(temp_path)
    os.startfile(temp_path)  # يشغّل الملف مباشرة على Windows
    # يمكن إضافة تأخير إذا أردت انتظار انتهاء الصوت قبل حذف الملف
    # import time; time.sleep(5)
    # os.remove(temp_path)  # إذا أردت حذف الملف بعد التشغيل

# البرنامج الرئيسي
def main():
    temp_audio_path = os.path.join(os.getcwd(), "temp_record.wav")
    record_audio(temp_audio_path, duration=6)
    text = speech_to_text(temp_audio_path)
    os.remove(temp_audio_path)
    
    print("📝 النص المستخرج:", text)

    if text == "" or text.isspace():
        response = "لم أسمعك جيدًا، حاول مرة أخرى من فضلك"
    else:
        answer = get_answer(text)
        if answer:
            response = answer
        else:
            response = "عذرًا، لم أتمكن من فهم سؤالك"

    print("🤖 الرد:", response)
    text_to_speech(response)

if __name__ == "__main__":
    while True:
        main()
        print("\n🔄 سيتم تسجيل سؤال جديد...")
