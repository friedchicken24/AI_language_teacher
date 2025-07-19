import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import google.generativeai as genai
import assemblyai as aai
import os
from gtts import gTTS
import av
import numpy as np
import uuid
import queue
from io import BytesIO
import wave

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Trợ lý ảo", page_icon="🤖", layout="wide")
st.title("🤖 Trợ Lý Ảo Thông Minh")

# --- CẤU HÌNH API KEYS VÀ CLIENTS ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ Không tìm thấy API Key: {e}. Vui lòng kiểm tra lại Secrets.")
    st.stop()

# --- CÁC HÀM TIỆN ÍCH (Không thay đổi) ---
def text_to_speech(text, lang='vi'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Lỗi khi tạo giọng nói: {e}")
        return None

def speech_to_text(wav_bytes):
    try:
        config = aai.TranscriptionConfig(language_code="vi")
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(wav_bytes)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Lỗi từ AssemblyAI: {transcript.error}")
            return None
        return transcript.text
    except Exception as e:
        st.error(f"Lỗi khi nhận diện giọng nói qua AssemblyAI: {e}")
        return None

def get_ai_response(user_text, conversation_history, system_prompt):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    messages = [{"role": "system", "content": system_prompt}]
    gemini_history = []
    for entry in conversation_history:
        role = 'model' if entry['role'] == 'assistant' else entry['role']
        gemini_history.append({'role': role, 'parts': [entry['content']]})
    chat = model.start_chat(history=gemini_history)
    try:
        response = chat.send_message(user_text)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi gọi Gemini: {e}")
        return "Tôi xin lỗi, tôi đang gặp sự cố với bộ não của mình."

# --- GIAO DIỆN VÀ LOGIC ---
with st.sidebar:
    st.title("Tùy Chọn Trợ Lý Ảo")
    personas = {
        "Trợ lý thân thiện": "Bạn là một trợ lý ảo tên là Zen...",
        "Nhà sử học uyên bác": "Bạn là một nhà sử học uyên bác...",
        "Chuyên gia công nghệ": "Bạn là một chuyên gia công nghệ hàng đầu..."
    }
    selected_persona_name = st.selectbox("Chọn một vai trò:", options=list(personas.keys()))
    system_prompt = personas[selected_persona_name]

# Khởi tạo session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "last_response_audio" not in st.session_state:
    st.session_state.last_response_audio = None
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = queue.Queue()

# Hiển thị lịch sử trò chuyện
chat_container = st.container()
with chat_container:
    for entry in st.session_state.conversation:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

# Tự động phát âm thanh
if st.session_state.last_response_audio:
    st.audio(st.session_state.last_response_audio, autoplay=True)
    if os.path.exists(st.session_state.last_response_audio):
        os.remove(st.session_state.last_response_audio)
    st.session_state.last_response_audio = None

# Component ghi âm
st.write("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Bảng điều khiển")
    # Sử dụng callback để ghi âm
    def audio_frame_callback(frame: av.AudioFrame):
        st.session_state.audio_buffer.put(frame.to_ndarray())

    webrtc_ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col2:
    st.subheader("Trạng thái")
    if webrtc_ctx.state.playing:
        st.success("🔴 Micro đang bật. Hãy nói đi!")
        if st.button("Dừng và gửi"):
            # Lấy dữ liệu từ queue
            frames = []
            while not st.session_state.audio_buffer.empty():
                frames.append(st.session_state.audio_buffer.get())

            if not frames:
                st.warning("Không có âm thanh nào được ghi lại. Vui lòng nói gần micro hơn.")
            else:
                st.info("Đã nhận được âm thanh. Đang xử lý...")
                
                # Ghép các frame và tạo file wav trong bộ nhớ
                sound_chunk = np.concatenate(frames, axis=1)
                sound_chunk = (sound_chunk * 32767).astype(np.int16)
                
                wav_buffer = BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(sound_chunk.tobytes())
                
                wav_bytes = wav_buffer.getvalue()

                # ---- BẮT ĐẦU LUỒNG XỬ LÝ ----
                with st.spinner("AI đang lắng nghe..."):
                    user_text = speech_to_text(wav_bytes)

                if user_text:
                    st.session_state.conversation.append({"role": "user", "content": user_text})
                    with st.spinner("AI đang suy nghĩ..."):
                        ai_response_text = get_ai_response(user_text, st.session_state.conversation, system_prompt)
                    
                    st.session_state.conversation.append({"role": "assistant", "content": ai_response_text})
                    
                    with st.spinner("AI đang chuẩn bị nói..."):
                        audio_file = text_to_speech(ai_response_text)
                    
                    if audio_file:
                        st.session_state.last_response_audio = audio_file
                    
                    st.rerun()
                else:
                    st.error("Không nhận diện được giọng nói. Hãy thử nói to và rõ hơn.")
    else:
        st.info("Nhấn 'Start' trên khung đen để cấp quyền và bắt đầu ghi âm.")
