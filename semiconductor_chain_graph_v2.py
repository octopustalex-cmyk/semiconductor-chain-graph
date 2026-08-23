import whisper

print("正在加载 Whisper large-v3...")

model = whisper.load_model("large-v3")

print("开始转写...")

audio_path = r"E:\Ritone\展会语音agent\录音\食物展览.m4a"

result = model.transcribe(
    audio_path,
    language="zh"
)

text = result["text"]

print("\n===== 转写结果 =====\n")
print(text)

with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("\n转写完成，结果已经保存到 transcription.txt")