import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import google.generativeai as genai
import assemblyai as aai
import os
from gtts import gTTS
import av
import numpy as np
import uuid
import queue # Dùng để giao tiếp giữa các luồng

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Trợ lý ảo", page_icon="🤖")
st.title("🤖 Trợ Lý Ảo Thông Minh")
st.sidebar.title("Tùy Chọn Trợ Lý Ảo")
st.sidebar.markdown("Chọn một 'cá tính' cho trợ lý ảo của chúng ta!")

# Lấy API keys
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ Không tìm thấy API Key: {e}. Vui lòng kiểm tra lại Secrets.")
    st.stop()

# --- CÁC HÀM TIỆN ÍCH (Không thay đổi) ---
def text_to_speech(text, lang='vi'):
    # ... (Giữ nguyên hàm này)
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Lỗi khi tạo giọng nói: {e}")
        return None

def speech_to_text(wav_bytes):
    # ... (Hàm này giờ nhận bytes thay vì list of frames)
    try:
        config = aai.TranscriptionConfig(language_code="vi") # Thêm language_code
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
    # ... (Giữ nguyên hàm này)
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

# --- CÁ TÍNH CỦA AI ---
personas = {
    "Trợ lý thân thiện": "Bạn là một trợ lý ảo tên là Zen...",
    "Nhà sử học uyên bác": "Bạn là một nhà sử học uyên bác...",
    "Chuyên gia công nghệ": "Bạn là một chuyên gia công nghệ hàng đầu..."
}
selected_persona_name = st.sidebar.selectbox("Chọn một vai trò:", options=list(personas.keys()))
system_prompt = personas[selected_persona_name]

# --- KHỞI TẠO STATE ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "last_response_audio" not in st.session_state:
    st.session_state.last_response_audio = None
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# --- GIAO DIỆN CHÍNH ---
st.write("---")

# Hiển thị lịch sử trò chuyện
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

# Tự động phát âm thanh nếu có
if st.session_state.last_response_audio:
    st.audio(st.session_state.last_response_audio, autoplay=True)
    if os.path.exists(st.session_state.last_response_audio):
        os.remove(st.session_state.last_response_audio)
    st.session_state.last_response_audio = None

# Component ghi âm và xử lý
# Sử dụng queue để nhận dữ liệu audio từ một luồng khác
audio_frames_queue = queue.Queue()

def audio_frame_callback(frame: av.AudioFrame):
    audio_frames_queue.put(frame.to_ndarray())

st.write("Nhấn 'Start' để bắt đầu ghi âm, 'Stop' để gửi.")
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
)

# Nút xử lý logic
if st.button("Xử lý giọng nói vừa ghi"):
    if not webrtc_ctx.state.playing:
        st.warning("Vui lòng nhấn 'Start' và ghi âm trước khi xử lý.")
    else:
        st.info("Đang xử lý...")
        
        # Lấy tất cả các frame từ queue
        audio_frames = []
        while not audio_frames_queue.empty():
            audio_frames.append(audio_frames_queue.get())
        
        if not audio_frames:
            st.warning("Không ghi âm được gì cả. Vui lòng nói vào micro.")
        else:
            # Ghép các frame lại và tạo file wav
            sample_rate = 48000
            sound_chunk = np.concatenate(audio_frames, axis=1)
            
            # Chuyển đổi sang định dạng 16-bit integer
            sound_chunk = (sound_chunk * 32767).astype(np.int16)

            # Tạo file wav trong bộ nhớ
            from io import BytesIO
            import wave

            wav_buffer = BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(sound_chunk.tobytes())
            
            st.session_state.audio_bytes = wav_buffer.getvalue()

            # Tải lại trang để bắt đầu chuỗi phản hồi
            st.rerun()

# --- LUỒNG PHẢN HỒI ---
if st.session_state.audio_bytes:
    with st.spinner("AI đang lắng nghe và suy nghĩ..."):
        # Lấy và xóa dữ liệu audio
        audio_to_process = st.session_state.audio_bytes
        st.session_state.audio_bytes = None

        # 1. TAI: Chuyển giọng nói thành văn bản
        user_text = speech_to_text(audio_to_process)
        
        if user_text:
            st.session_state.conversation.append({"role": "user", "content": user_text})
            
            # 2. NÃO: Lấy câu trả lời
            ai_response_text = get_ai_response(user_text, st.session_state.conversation, system_prompt)
            st.session_state.conversation.append({"role": "assistant", "content": ai_response_text})

            # 3. MIỆNG: Tạo file âm thanh
            audio_file = text_to_speech(ai_response_text)
            if audio_file:
                st.session_state.last_response_audio = audio_file
            
            # Tải lại lần cuối để hiển thị kết quả và phát âm thanh
            st.rerun()
        else:
            st.error("Không thể nhận diện giọng nói. Vui lòng thử lại.")
            st.rerun() # Tải lại để xóa spinner
