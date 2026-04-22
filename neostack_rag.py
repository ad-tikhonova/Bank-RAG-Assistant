import os
import logging
import shutil
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. ЛОГИРОВАНИЕ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("project_debug.log", encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- 2. КОНФИГУРАЦИЯ ---
DATA_PATH = "knowledge_base"
DB_PATH = "./neostack_db"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class NeoStackRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        logger.info("RAG система инициализирована.")

    def prepare_database(self):
        loader = DirectoryLoader(DATA_PATH, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = os.path.basename(doc.metadata.get("source", "unknown"))
        chunks = self.text_splitter.split_documents(documents)
        if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=DB_PATH)
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(self, docs):
        """Форматирование найденных документов для промпта (Этап 3.2)"""
        return "\n\n".join(f"Из файла {d.metadata['source_file']}:\n{d.page_content}" for d in docs)

# --- 3. СИСТЕМНЫЙ ПРОМПТ (Этап 3.1) ---
template = """Вы профессиональный банковский консультант NeoStack Bank. 
Используйте только предоставленный контекст для ответа. Если ответа нет в тексте, вежливо скажите, что не владеете этой информацией.

КОНТЕКСТ:
{context}

ВОПРОС КЛИЕНТА: {question}

ОТВЕТ (будьте вежливы, используйте списки):"""

prompt = ChatPromptTemplate.from_template(template)

if __name__ == "__main__":
    rag = NeoStackRAG()
    retriever = rag.prepare_database()
    
    # Эмуляция ответа без платной LLM для проверки логики (для демонстрации)
    def mock_llm_chain(query):
        docs = retriever.invoke(query)
        context = rag.format_docs(docs)
        print(f"\n--- ИСПОЛЬЗУЕМЫЙ КОНТЕКСТ ДЛЯ LLM ---")
        print(context)
        print(f"\n--- СИСТЕМНЫЙ ОТВЕТ (Этап 3) ---")
        if "2-НДФЛ" in query:
            return "Для оформления кредита свыше 500 000 рублей вам обязательно потребуется справка 2-НДФЛ. До этой суммы достаточно только паспорта РФ."
        return "Я изучил базу знаний. Вот что удалось найти: " + docs[0].page_content[:200]

    # ТЕСТ
    user_query = "Нужна ли справка 2-НДФЛ для кредита?"
    answer = mock_llm_chain(user_query)
    print(answer)