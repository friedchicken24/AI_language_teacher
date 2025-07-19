import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import google.generativeai as genai
import assemblyai as aai
import os
from gtts import gTTS
import av
import numpy as np
import uuid

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

# --- LỚP XỬ LÝ ÂM THANH ---
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames_buffer = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Chỉ cần lưu frame vào buffer
        self.frames_buffer.append(frame.to_ndarray())
        return frame

# --- CÁC HÀM TIỆN ÍCH ---
# (Các hàm text_to_speech, speech_to_text, get_ai_response giữ nguyên như code cũ của bạn)
# ... Dán các hàm đó vào đây ...
def text_to_speech(text, lang='vi'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = f"response_{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        st.error(f"Lỗi khi tạo giọng nói: {e}")
        return None

def speech_to_text(audio_frames, sample_rate):
    try:
        sound_chunk = np.concatenate(audio_frames, axis=1)
        output_filename = f"input_{uuid.uuid4()}.wav"
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
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(output_filename)
        os.remove(output_filename)
        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Lỗi từ AssemblyAI: {transcript.error}")
            return None
        else:
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

# --- CÁ TÍNH CỦA AI ---
personas = {
    "Trợ lý thân thiện": "Bạn là một trợ lý ảo tên là Zen, rất thân thiện, tích cực và luôn sẵn lòng giúp đỡ. Hãy trả lời bằng tiếng Việt.",
    "Nhà sử học uyên bác": "Bạn là một nhà sử học uyên bác. Hãy trả lời mọi câu hỏi với giọng điệu trang trọng, đưa ra các chi tiết và bối cảnh lịch sử thú vị. Hãy trả lời bằng tiếng Việt.",
    "Chuyên gia công nghệ": "Bạn là một chuyên gia công nghệ hàng đầu. Hãy giải thích các khái niệm phức tạp một cách đơn giản, đưa ra các ví dụ thực tế và luôn cập nhật các xu hướng mới nhất. Hãy trả lời bằng tiếng Việt."
}
selected_persona_name = st.sidebar.selectbox("Chọn một vai trò:", options=list(personas.keys()))
system_prompt = personas[selected_persona_name]

# --- KHỞI TẠO STATE ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = None
if "last_response_audio" not in st.session_state:
    st.session_state.last_response_audio = None

# --- GIAO DIỆN CHÍNH ---

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

# Component ghi âm
webrtc_ctx = webrtc_streamer(
    key="recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"video": False, "audio": True},
)

st.write("---")
# Nút bấm được đặt bên ngoài component
if not webrtc_ctx.state.playing:
    st.info("Nhấn 'Start' trên khung đen ở trên để cấp quyền micro và bắt đầu.")
else:
    if st.button("Dừng lại và Gửi câu hỏi"):
        if webrtc_ctx.audio_processor:
            # Lấy dữ liệu âm thanh từ buffer của audio_processor
            st.session_state.audio_buffer = webrtc_ctx.audio_processor.frames_buffer.copy()
            webrtc_ctx.audio_processor.frames_buffer.clear() # Dọn dẹp buffer
        
        # Tải lại trang để bắt đầu xử lý
        st.rerun()

# --- LUỒNG XỬ LÝ ÂM THANH ---
# Chỉ xử lý khi có dữ liệu trong buffer
if st.session_state.audio_buffer:
    with st.spinner("Đang xử lý âm thanh..."):
        audio_frames = st.session_state.audio_buffer
        st.session_state.audio_buffer = None # Xóa buffer sau khi lấy

        # 1. TAI: Chuyển giọng nói thành văn bản
        sample_rate = 48000
        user_text = speech_to_text(audio_frames, sample_rate)

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
