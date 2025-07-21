# final_teacher.py - PHIÊN BẢN SỬA LỖI CUỐI CÙNG (Whisper + Phản hồi âm thanh)
import os
import soundfile as sf
import sounddevice as sd    
from groq import Groq
from TTS.api import TTS
import numpy as np
import whisper

# --- PHẦN 1: CÀI ĐẶT CÁC DỊCH VỤ  ---

# 1.1. Cài đặt "Bộ Não" - Groq
try:
    # NHỚ THAY API KEY CỦA BẠN VÀO ĐÂY
    groq_client = Groq(api_key="")
    print("✅ Kết nối tới Groq (Bộ não) thành công.")
except Exception as e:
    print(f"❌ LỖI: Không thể kết nối tới Groq. {e}")
    exit()

# 1.2. Cài đặt "Giọng Nói" - Coqui TTS
try:
    print("   Đang tải mô hình giọng nói... (Có thể mất vài phút)")
    tts_model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True).to(device)
    print("✅ Mô hình giọng nói đã sẵn sàng.")
except Exception as e:
    print(f"❌ LỖI: Không thể tải mô hình TTS. {e}")
    exit()

# 1.3. Cài đặt "Đôi Tai" - Whisper và Microphone
try:
    print("   Đang tải mô hình Whisper... (Có thể mất vài phút)")
    whisper_model = whisper.load_model("base")
    print("✅ Mô hình Whisper đã sẵn sàng.")
    print("   Đảm bảo microphone đã được kết nối và hoạt động.")
    print("   Bạn có thể nói tiếng Anh hoặc tiếng Đức.")
except Exception as e:
    print(f"❌ LỖI: Không thể tải mô hình Whisper. {e}")
    exit()

# --- PHẦN 2: THIẾT LẬP VAI TRÒ GIÁO VIÊN ---
system_prompt = """
You are a multilingual language teacher named Alex. You are an expert in both English and German.
Your student will practice with you. Your main goal is to be helpful and encouraging.
- When the student writes in English, respond as an English teacher.
- When the student writes in German, respond as a German teacher.
- If the student makes a mistake in either language, gently correct it. First, show the corrected sentence. Then, briefly and simply explain the mistake in Vietnamese.
- Always be positive and keep the conversation natural.
"""
conversation_history = [{"role": "system", "content": system_prompt}]

# --- PHẦN 3: MỘT VÒNG LẶP DUY NHẤT ---

print("\n--- Tui là Đức Anh - Giáo viên ngôn ngữ (Dức - Anh) của bạn ---")
print("Nhấn Enter, nói chuyện với tui, rồi chờ kết quả nhé. Nói 'quit' hoặc 'thoát' để dừng lại.")
print("-" * 40)

while True:
    input("=> Nhấn Enter để bắt đầu nói...")
    print("🎤 Bồ nói đi, Tui đang nghe nè...")

    # Ghi âm trực tiếp bằng sounddevice
    duration = 5  # số giây ghi âm, có thể chỉnh tùy ý
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("temp.wav", audio, fs)

    print("🤔 Chờ tui xíu nha, đang xử lý...")

    # Nhận diện bằng Whisper
    try:
        result = whisper_model.transcribe("temp.wav", language=None)
        user_text = result["text"].strip()

        if not user_text:
            print("👂 Hình như bồ chưa nói gì cả. Thử lại nhé.")
            continue

        print(f"Bồ nói: {user_text}")
    except Exception as e:
        print(f"Lỗi khi nhận dạng giọng nói: {e}")
        continue

    # 3. Xử lý logic trò chuyện
    if "quit" in user_text.lower() or "thoát" in user_text.lower():
        break

    conversation_history.append({"role": "user", "content": user_text})

    try:
        # 4. Groq tạo ra câu trả lời
        completion = groq_client.chat.completions.create(
            model="kimi-k2", 
            messages=conversation_history
        )
        ai_response_text = completion.choices[0].message.content
        
        print(f"Đức Anh (text): {ai_response_text}")
        conversation_history.append({"role": "assistant", "content": ai_response_text})

        # ==================================================================
        # PHẦN PHẢN HỒI ÂM THANH CỦA AI 
        # ==================================================================
        print("Đức Anh (voice): ...speaking...")
        
        # 5. TTS chuyển văn bản thành dữ liệu âm thanh
        #    Hàm tts() trả về một list các số float đại diện cho sóng âm.
        wav_data = tts_model.tts(text=ai_response_text)
        
        # 6. Dùng sounddevice để phát trực tiếp
        sample_rate = getattr(tts_model, "output_sample_rate", 22050)
        sd.play(np.array(wav_data), sample_rate)
        sd.wait() # Chờ cho đến khi phát xong
        # ==================================================================

    except Exception as e:
        print(f"❌ An error occurred during conversation: {e}")
        conversation_history.pop()

# Lời chào tạm biệt cuối cùng
final_goodbye_text = "Tạm biệt bồ nha! Hẹn gặp lại lần sau nhé."
print(f"\nĐức Anh (text): {final_goodbye_text}")
wav_data = tts_model.tts(text=final_goodbye_text)
sample_rate = getattr(tts_model, "output_sample_rate", 22050)
sd.play(np.array(wav_data), sample_rate)
sd.wait()