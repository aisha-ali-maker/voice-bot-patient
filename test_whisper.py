import whisper

# تحميل نموذج Whisper
model = whisper.load_model("base")

# تحويل الصوت إلى نص
result = model.transcribe("test.mp3")

# عرض النص المستخرج
print("النص المكتشف:")
print(result["text"])
