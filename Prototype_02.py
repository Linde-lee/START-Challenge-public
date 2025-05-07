import streamlit as st
import openai
import os
import pdfplumber
import tempfile

# ========== CONFIGURATION ========== #
openai.api_key = os.getenv("OPENAI_API_KEY") or "sk-your-openai-key"  # <-- Replace for local testing

st.set_page_config(page_title="📄 PDF 到 中文解释", layout="centered")
st.title("📄 德文PDF ➜ 简体中文解释 + 聊天机器人")

# ========== FILE UPLOAD ========== #
st.markdown("请上传包含德文内容的PDF文件")
pdf_file = st.file_uploader("上传医院账单 PDF", type=["pdf"])

extracted_text = ""
translated_text = ""

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() + "\n"

    st.markdown("### ✅ 提取到的德文内容:")
    st.text_area("原始德文文本:", extracted_text, height=200)

    # ========== TRANSLATION + SIMPLIFICATION ========== #
    if st.button("➡️ 翻译并简化为中文"):
        with st.spinner("正在翻译和简化，请稍候..."):
            try:
                system_prompt = (
                    "你是一位语言助手，能将德文官文翻译成简体中文，并用简单、易懂的语言解释，适合12岁儿童理解。"
                )
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"请将以下德文翻译并解释为简体中文：\n{extracted_text}"}
                    ],
                    temperature=0.5
                )
                translated_text = response.choices[0].message['content']
                st.success("✅ 翻译完成")
                st.markdown("### ✨ 简体中文解释:")
                st.text_area("翻译内容:", translated_text, height=200)
                st.session_state.translated_text = translated_text
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

# ========== CHATBOT SECTION ========== #
st.markdown("---")
st.markdown("### 💬 中文聊天助手")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": "你是一个中文助手，帮助用户理解上传的医院账单内容，用简体中文回答问题，语言要通俗易懂。"}
    ]

user_question = st.text_input("你有什么问题？")
if st.button("发送问题") and user_question:
    if "translated_text" in st.session_state:
        st.session_state.chat_messages.append({"role": "user", "content": f"这是账单内容：\n{st.session_state.translated_text}"})
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=st.session_state.chat_messages,
                temperature=0.5
            )
            reply = response.choices[0].message["content"]
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.markdown(f"**回答：** {reply}")
        except Exception as e:
            st.error(f"发生错误：{str(e)}")
    else:
        st.warning("请先上传并翻译PDF")

# ========== SHOW CHAT HISTORY ========== #
if st.checkbox("显示聊天记录"):
    for msg in st.session_state.chat_messages[1:]:
        role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
        st.markdown(f"**{role}:** {msg['content']}")
