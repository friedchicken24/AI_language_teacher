# test_voice.py
import torch
from TTS.api import TTS
import os

# In ra thư mục làm việc hiện tại để kiểm tra
print(f"Thư mục làm việc hiện tại: {os.getcwd()}")

# Văn bản bạn muốn AI đọc
text_to_speak = "Hello, my name is Alex. I will be your English teacher today. Let's start our first lesson."

# Chọn thiết bị: GPU nếu có, không thì CPU
device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Sẽ sử dụng thiết bị: {device}")

print("Đang tải mô hình TTS... (Lần đầu có thể mất vài phút để tải về)")

try:
    # Tải một mô hình tiếng Anh có sẵn, chất lượng cao từ Coqui
    # Nó sẽ tự động tải các file cần thiết về và lưu vào cache
    tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True).to(device)
    print("✅ Mô hình đã được tải thành công.")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    
    print("   Vui lòng kiểm tra kết nối mạng và đảm bảo Coqui-TTS đã được cài đặt đúng.")
    exit()

# Tên file âm thanh đầu ra
output_file = "teacher_alex_voice.wav"
print(f"\nĐang tạo giọng nói và lưu vào file '{output_file}'...")

try:
    # Sử dụng mô hình để tạo giọng nói và lưu ra file
    tts.tts_to_file(text=text_to_speak, file_path=output_file)
    print(f"\n✅ HOÀN THÀNH! Hãy kiểm tra file '{output_file}' trong thư mục dự án của bạn.")
except Exception as e:
    print(f"❌ Lỗi khi tạo file âm thanh: {e}")