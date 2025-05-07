import streamlit as st
import os
import pdfplumber
import tempfile
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from haystack.nodes import FARMReader, TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor

# ========== SETUP MODELS ========== #
@st.cache_resource
def load_translation_pipeline():
    model_name = "Helsinki-NLP/opus-mt-de-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer, src_lang="de", tgt_lang="zh")

@st.cache_resource
def load_qa_pipeline():
    document_store = InMemoryDocumentStore()
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    return ExtractiveQAPipeline(reader=reader, retriever=None), document_store

translation_pipeline = load_translation_pipeline()
qa_pipeline, doc_store = load_qa_pipeline()

# ========== PAGE SETUP ========== #
st.set_page_config(page_title="📄 PDF 到 中文解释", layout="centered")
st.title("📄 德文PDF ➜ 中文翻译 + 本地聊天机器人")

# ========== FILE UPLOAD ========== #
st.markdown("请上传包含德文内容的PDF文件")
pdf_file = st.file_uploader("上传医院账单 PDF", type=["pdf"])

extracted_text = ""
if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text() + "\n"

    st.markdown("### ✅ 提取到的德文内容:")
    st.text_area("原始德文文本:", extracted_text, height=200)

    if st.button("➡️ 翻译为中文"):
        with st.spinner("正在翻译..."):
            try:
                chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]
                translated = [translation_pipeline(chunk)[0]['translation_text'] for chunk in chunks]
                final_translation = "\n".join(translated)
                st.session_state.chat_context = final_translation
                st.success("✅ 翻译完成")
                st.text_area("简体中文翻译:", final_translation, height=200)

                # For chatbot: load into Haystack doc store
                doc_store.write_documents([{"content": final_translation, "meta": {"name": "hospital_bill"}}])
            except Exception as e:
                st.error(f"翻译错误: {str(e)}")

# ========== QA CHATBOT ========== #
st.markdown("---")
st.markdown("### 💬 中文聊天助手 (使用 Haystack 本地问答系统)")
user_question = st.text_input("你有什么问题？")
if st.button("发送问题") and user_question:
    if 'chat_context' not in st.session_state:
        st.warning("请先上传并翻译PDF")
    else:
        with st.spinner("正在查找答案..."):
            try:
                prediction = qa_pipeline.run(
                    query=user_question,
                    documents=doc_store.get_all_documents(),
                    params={"Reader": {"top_k": 1}}
                )
                answer = prediction['answers'][0].answer if prediction['answers'] else "对不起，我没有找到答案。"
                st.markdown(f"**回答：** {answer}")
            except Exception as e:
                st.error(f"错误: {str(e)}")
