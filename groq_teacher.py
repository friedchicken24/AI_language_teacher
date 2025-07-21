# groq_teacher.py

from groq import Groq

# --- PHẦN CẤU HÌNH ---
# Dán API Key của Groq (bắt đầu bằng gsk_...) vào đây.
try:
    api_key = "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"
    client = Groq(api_key=api_key)
    print("✅ Kết nối tới Groq thành công.")
except Exception as e:
    print(f"❌ LỖI: Không thể kết nối tới Groq. Vui lòng kiểm tra lại API Key.")
    print(f"   Chi tiết lỗi: {e}")
    exit()

system_prompt = """
You are a multilingual language teacher named Alex. You are an expert in both English and German.
Your student will practice with you. Your main goal is to be helpful and encouraging.

- When the student writes in English, respond as an English teacher.
- When the student writes in German, respond as a German teacher.
- If the student makes a mistake in either language, gently correct it. First, show the corrected sentence. Then, briefly and simply explain the mistake in Vietnamese.
- Always be positive and keep the conversation natural.
"""
# Thiết lập lời nhắc hệ thống cho AI
client.chat.system_prompt = system_prompt

# Lưu trữ lịch sử cuộc trò chuyện
conversation_history = [{"role": "system", "content": system_prompt}]

#  GIAO DIỆN DÒNG LỆNH 
print("\n--- AI English Teacher (Alex on Groq) ---")
print("Let's practice English! You can start the conversation.")
print("Type 'quit' to end the session.")
print("-" * 40)

# Vòng lặp trò chuyện
while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("Alex: Tạm biệt bồ nha. Hẹn gặp lại <3.")
        break

    # Thêm tin nhắn của người dùng vào lịch sử
    conversation_history.append({"role": "user", "content": user_input})

    try:
        # Gửi toàn bộ lịch sử cuộc trò chuyện đến API của Groq
        completion = client.chat.completions.create(
            # Sử dụng mô hình Llama 3 8B, rất nhanh và thông minh
            model="llama3-8b-8192",
            messages=conversation_history,
            temperature=0.7, # Độ "sáng tạo" của câu trả lời
            max_tokens=1024, # Giới hạn độ dài câu trả lời
        )

        # Lấy câu trả lời của AI
        ai_response = completion.choices[0].message.content

        # In câu trả lời ra màn hình
        print(f"Alex: {ai_response}")

        # Thêm câu trả lời của AI vào lịch sử để nó "nhớ"
        conversation_history.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # Xóa tin nhắn cuối cùng của người dùng khỏi lịch sử nếu có lỗi
        conversation_history.pop()
        print("   Please try again.")