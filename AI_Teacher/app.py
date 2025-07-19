
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import openai
import os
from gtts import gTTS
import av
import numpy as np
import time
import uuid

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Trợ lý ảo", page_icon="🤖")
st.title("🤖 Trợ Lý Ảo Thông Minh")
st.write("Nói chuyện với tôi nhé! Tôi đang lắng nghe...")

# Lấy API key
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ Vui lòng thêm OpenAI API key vào Secrets của ứng dụng.")
    st.stop()

# --- LỚP XỬ LÝ ÂM THANH ---
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self._frames = []
        self.start_recording = False
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.start_recording:
            self._frames.append(frame.to_ndarray())
        return frame
    @property
    def frames(self):
        return self._frames
    def clear_frames(self):
        self._frames = []

# --- CÁC HÀM TIỆN ÍCH ---
def text_to_speech(text, lang='vi'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        # Tạo tên file duy nhất để tránh xung đột
        filename = f"response_{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Lỗi khi tạo giọng nói: {e}")
        return None

def speech_to_text(audio_data, sample_rate):
    try:
        # Lưu file wav tạm thời
        output_filename = f"input_{uuid.uuid4()}.wav"
        sound_chunk = np.concatenate(audio_data, axis=1)
        # Viết header cho file WAV
        with open(output_filename, "wb") as f:
            f.write(b"RIFF")
            f.write(b"\x00\x00\x00\x00")
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(b"\x10\x00\x00\x00")
            f.write(b"\x01\x00\x01\x00")
            f.write(sample_rate.to_bytes(4, "little"))
            f.write((sample_rate * 2).to_bytes(4, "little"))
            f.write(b"\x02\x00\x10\x00")
            f.write(b"data")
            f.write(len(sound_chunk.tobytes()).to_bytes(4, "little"))
            f.write(sound_chunk.tobytes())
        
        # Gửi đến Whisper
        with open(output_filename, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        os.remove(output_filename) # Xóa file tạm
        return transcript.text
    except Exception as e:
        st.error(f"Lỗi khi nhận diện giọng nói: {e}")
        return None
        
def get_ai_response(user_text, conversation_history):
    messages = [{"role": "system", "content": "Bạn là một trợ lý ảo thân thiện, thông minh và nói chuyện bằng tiếng Việt. Hãy trả lời ngắn gọn và tự nhiên."}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})
    
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=messages)
        ai_text = response.choices[0].message.content
        return ai_text
    except Exception as e:
        st.error(f"Lỗi khi gọi GPT: {e}")
        return "Tôi xin lỗi, tôi đang gặp sự cố."

# --- KHỞI TẠO STATE CỦA ỨNG DỤNG ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# Hiển thị lịch sử trò chuyện
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

# --- BỘ ĐIỀU KHIỂN GHI ÂM ---
# Sử dụng cột để sắp xếp các nút
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.is_recording:
        if st.button("🎤 Bắt đầu nói"):
            st.session_state.is_recording = True
            st.rerun()
    else:
        if st.button("🛑 Dừng lại và Gửi"):
            st.session_state.is_recording = False
            st.rerun()

# --- LUỒNG XỬ LÝ CHÍNH ---
if st.session_state.is_recording:
    st.info("🔴 Đang nghe... (Nhấn 'Dừng lại' khi bạn nói xong)")
    webrtc_ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"video": False, "audio": True},
    )
    if webrtc_ctx.audio_processor:
        st.session_state.audio_frames = webrtc_ctx.audio_processor.frames

# Chỉ xử lý khi đã dừng ghi âm và có dữ liệu
if not st.session_state.is_recording and st.session_state.audio_frames:
    with st.spinner("Đang xử lý..."):
        # 1. TAI: Chuyển giọng nói thành văn bản
        sample_rate = 48000 # Tần số mẫu mặc định của webrtc
        user_text = speech_to_text(st.session_state.audio_frames, sample_rate)
        st.session_state.audio_frames = [] # Xóa dữ liệu cũ

        if user_text:
            # Cập nhật lịch sử trò chuyện với lời của người dùng
            st.session_state.conversation.append({"role": "user", "content": user_text})
            
            # 2. NÃO: Lấy câu trả lời từ AI
            ai_response_text = get_ai_response(user_text, st.session_state.conversation)
            
            # Cập nhật lịch sử trò chuyện với lời của AI
            st.session_state.conversation.append({"role": "assistant", "content": ai_response_text})

            # 3. MIỆNG: Chuyển câu trả lời của AI thành giọng nói và phát
            audio_file = text_to_speech(ai_response_text)
            if audio_file:
                # Tự động phát âm thanh và xóa file
                st.audio(audio_file, autoplay=True)
                time.sleep(1) # Chờ một chút để chắc chắn st.audio đã load
                os.remove(audio_file)
            
            # Tải lại trang để hiển thị cuộc hội thoại mới
            st.rerun()
