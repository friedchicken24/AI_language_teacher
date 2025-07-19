import streamlit as st
import google.generativeai as genai
import assemblyai as aai
import os
from gtts import gTTS
import uuid
from io import BytesIO
import librosa
import numpy as np
import soundfile as sf

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(page_title="Trợ lý ảo", page_icon="🤖", layout="wide")
st.title("🤖 Trợ Lý Ảo Thông Minh")

# --- CẤU HÌNH API KEYS ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    aai.settings.api_key = st.secrets["ASSEMBLYAI_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ Không tìm thấy API Key: {e}. Vui lòng kiểm tra lại mục 'Secrets'.")
    st.stop()

# --- CÁC HÀM TIỆN ÍCH ---
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

# --- GIAO DIỆN ---
with st.sidebar:
    st.title("Tùy Chọn Trợ Lý Ảo")
    personas = {
        "Trợ lý thân thiện": "Bạn là một trợ lý ảo tên là Zen...",
        "Nhà sử học uyên bác": "Bạn là một nhà sử học uyên bác...",
        "Chuyên gia công nghệ": "Bạn là một chuyên gia công nghệ hàng đầu..."
    }
    selected_persona_name = st.selectbox("Chọn vai trò:", options=list(personas.keys()))
    system_prompt = personas[selected_persona_name]

# Khởi tạo session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "last_response_audio" not in st.session_state:
    st.session_state.last_response_audio = None

# Hiển thị lịch sử trò chuyện
chat_container = st.container(height=400)
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

# --- KHU VỰC TẢI FILE LÊN ---
st.divider()
st.subheader("Trò chuyện với AI")
st.write("Ghi âm một câu hỏi bằng điện thoại hoặc máy tính, sau đó tải file lên đây.")

uploaded_file = st.file_uploader("Tải file âm thanh của bạn (MP3, WAV, M4A)...", type=['mp3', 'wav', 'm4a'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Gửi âm thanh để AI xử lý"):
        with st.spinner("Đang xử lý âm thanh của bạn..."):
            # Đọc file người dùng tải lên
            audio_bytes = uploaded_file.getvalue()
            
            # Lưu vào bộ nhớ để librosa đọc
            audio_buffer = BytesIO(audio_bytes)
            
            # Resample âm thanh về 16kHz
            y, sr = librosa.load(audio_buffer, sr=None)
            target_sr = 16000
            if sr != target_sr:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
            
            # Chuyển đổi lại thành bytes định dạng WAV
            wav_processed_buffer = BytesIO()
            sf.write(wav_processed_buffer, y, target_sr, format='WAV', subtype='PCM_16')
            wav_bytes_processed = wav_processed_buffer.getvalue()

        # ---- BẮT ĐẦU LUỒNG XỬ LÝ ----
        with st.spinner("AI đang lắng nghe..."):
            user_text = speech_to_text(wav_bytes_processed)

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
            st.error("Không nhận diện được giọng nói. Hãy thử ghi âm lại rõ hơn.")
