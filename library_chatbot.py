import os
import sys
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

# ================================
# 1. ❌ Windows에서는 pysqlite3 패치 제거!
# ================================
# __import__('pysqlite3')
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ================================
# 2. LangChain & Embedding & Chroma
# ================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

from langchain_chroma import Chroma


# ================================
# 3. Gemini API 키 설정
# ================================
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()


# ================================
# 4. PDF 경로 및 Chroma 폴더명
# ================================
PDF_PATH = r"/mount/src/librarychatbot_gemini/인천 섬 갯벌에 대한 생태적 가치화 방안과 적용.pdf"
PDF_NAME = os.path.splitext(os.path.basename(PDF_PATH))[0]
VECTOR_DIR = f"./chroma_db_{PDF_NAME}"


# ================================
# 5. Streamlit 캐시 초기화 버튼
# ================================
if st.button("🔄 캐시 및 임베딩 데이터 초기화"):
    import shutil
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
        st.success("🗑️ 기존 ChromaDB 데이터 삭제 완료!")
    st.cache_resource.clear()
    st.success("♻️ Streamlit 캐시 초기화 완료! 앱을 새로고침하세요.")


# ================================
# 6. PDF load & split
# ================================
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


# ================================
# 7. ChromaDB 생성 함수
# ================================
client_settings = {
    "chroma_tenant": "default_tenant",
    "chroma_collection": "default"
}

@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개 청크로 분할했습니다.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 중...")

    vectorstore = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
        collection_name="default"
    )

    st.success("💾 새로운 ChromaDB 저장 완료!")
    return vectorstore


# ================================
# 8. 기존 DB 로드 or 생성
# ================================
@st.cache_resource
def get_vectorstore(_docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(VECTOR_DIR):
        st.info("📂 기존 ChromaDB 로드 중...")
        return Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="default",
            client_settings={ 
                "chroma_tenant": "default_tenant",
                "chroma_collection": "default"
            }
        )
    else:
        return create_vector_store(_docs)


# ================================
# 9. RAG 체인 초기화
# ================================
@st.cache_resource
def initialize_components(selected_model):
    pages = load_and_split_pdf(PDF_PATH)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # ---- 질문 재구성 프롬프트 ----
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might 
    reference context in the chat history, formulate a standalone 
    question. Do NOT answer the question.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    # ---- QA 프롬프트 ----
    qa_system_prompt = """
    You are an assistant for question-answering tasks.
    Use the retrieved context to answer the question.
    If you don’t know the answer, just say so.
    Answer in Korean with natural emojis.
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# ================================
# 10. UI
# ================================
st.header("인천 섬 갯벌에 대한 생태적 가치화 방안과 적용")

if not os.path.exists(VECTOR_DIR):
    st.info("🔄 첫 실행입니다. PDF를 임베딩 중입니다...")
else:
    st.info(f"📂 '{PDF_NAME}' 벡터DB 로드 완료")


option = st.selectbox(
    "Select Gemini Model",
    ("gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite"),
    help="Gemini 2.0 Flash 추천!"
)

try:
    with st.spinner("챗봇 초기화 중..."):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇 준비 완료!")
except Exception as e:
    st.error(f"⚠️ 초기화 오류: {str(e)}")
    st.stop()


# ================================
# 11. 대화 히스토리 저장
# ================================
chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)


# ================================
# 12. 기존 메시지 출력
# ================================
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


# ================================
# 13. 사용자 입력 처리
# ================================
if user_input := st.chat_input("질문을 입력하세요..."):
    st.chat_message("human").write(user_input)

    with st.chat_message("ai"):
        with st.spinner("답변 생성 중..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "safe_sea_chat"}}
            )

            answer = response["answer"]
            st.write(answer)

            with st.expander("📘 참고 문서"):
                for doc in response["context"]:
                    st.markdown(doc.metadata.get("source", "문서 출처 없음"))
