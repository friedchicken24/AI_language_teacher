
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import google.generativeai as genai
import assemblyai as aai
from google.oauth2 import service_account
import json
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
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]) 
except KeyError:
    st.error("⚠️ Vui lòng thêm GOOGLE_API_KEY vào Secrets của ứng dụng.")
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

# Cấu hình AssemblyAI
try:
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError:
    st.error("⚠️ Vui lòng thêm ASSEMBLYAI_API_KEY vào Secrets.")
    st.stop()

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

        # Gửi đến AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(output_filename)

        os.remove(output_filename) # Xóa file tạm

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Lỗi từ AssemblyAI: {transcript.error}")
            return None
        else:
            return transcript.text
    except Exception as e:
        st.error(f"Lỗi khi nhận diện giọng nói qua AssemblyAI: {e}")
        return None
        
def get_ai_response(user_text, conversation_history):
    # Khởi tạo model Gemini
    model = genai.GenerativeModel('gemini-1.5-flash-latest') 
    
    gemini_history = []
    for entry in conversation_history:
        # Gemini dùng 'model' cho vai trò của AI
        role = 'model' if entry['role'] == 'assistant' else entry['role']
        gemini_history.append({'role': role, 'parts': [entry['content']]})

    # Bắt đầu một phiên chat với lịch sử cũ
    chat = model.start_chat(history=gemini_history)
    
    try:
        # Gửi tin nhắn mới của người dùng
        response = chat.send_message(user_text)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini: {e}")
        return "Tôi xin lỗi, tôi đang gặp sự cố với bộ não của mình."

# KHỞI TẠO STATE 
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# lịch sử trò chuyện
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

if "last_response_audio" not in st.session_state:
    st.session_state.last_response_audio = None

if st.session_state.last_response_audio:
    st.audio(st.session_state.last_response_audio, autoplay=True)
    # Xóa file sau khi đã thêm vào widget để tránh đầy bộ nhớ server
    if os.path.exists(st.session_state.last_response_audio):
        os.remove(st.session_state.last_response_audio)
    # Reset state để không phát lại ở lần rerun tiếp theo
    st.session_state.last_response_audio = None

# BỘ ĐIỀU KHIỂN GHI ÂM
# Sử dụng cột để sắp xếp các nút
col1, col2 = st.columns(2)
if "process_audio" not in st.session_state:
    st.session_state.process_audio = False
with col1:
    if not st.session_state.is_recording:
        if st.button("🎤 Bắt đầu nói"):
            st.session_state.is_recording = True
            st.rerun()
     if st.button("🛑 Dừng lại và Gửi"):
            st.session_state.is_recording = False
            # Đặt cờ báo hiệu rằng cần xử lý âm thanh
            st.session_state.process_audio = True 

# LUỒNG XỬ LÝ 9
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

else:
    # Nếu không ghi âm, hiển thị thông báo
    st.info("Nhấn 'Bắt đầu nói' để trò chuyện với tôi.")


# Chỉ xử lý khi đã dừng ghi âm và có dữ liệu
if st.session_state.process_audio:
    if not st.session_state.audio_frames:
        st.warning("Không ghi âm được gì cả. Vui lòng thử lại.")
        st.session_state.process_audio = False # Reset cờ
    else:
        with st.spinner("Đang xử lý..."):
            # Lấy dữ liệu âm thanh đã lưu
            audio_frames_to_process = st.session_state.audio_frames.copy()
            # Xóa dữ liệu cũ trong state ngay lập tức
            st.session_state.audio_frames = [] 
       
        
     # 1. TAI: Chuyển giọng nói thành văn bản
            sample_rate = 48000
            user_text = speech_to_text(audio_frames_to_process, sample_rate)

            if user_text:
                # Cập nhật lịch sử trò chuyện với lời của người dùng
                st.session_state.conversation.append({"role": "user", "content": user_text})
                
                # 2. NÃO: Lấy câu trả lời từ AI
                ai_response_text = get_ai_response(user_text, st.session_state.conversation, system_prompt)
                
                # Cập nhật lịch sử trò chuyện 
                st.session_state.conversation.append({"role": "assistant", "content": ai_response_text})

                # 3. MIỆNG: Chuyển câu trả lời của AI thành giọng nói và phát
                audio_file = text_to_speech(ai_response_text)
                if audio_file:
                    st.session_state.last_response_audio = audio_file # Lưu tên file để phát ở lần chạy sau

        # Reset cờ xử lý sau khi đã xong
        st.session_state.process_audio = False
        # Tải lại trang MỘT LẦN DUY NHẤT sau khi đã xử lý xong
        st.rerun()
