import os
import logging
import gradio as gr
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. ЛОГИРОВАНИЕ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("project_debug.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. КОНФИГУРАЦИЯ ---
DATA_PATH = "knowledge_base"
DB_PATH = "./neostack_db"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Инициализируем эмбеддинги
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def run_indexing():
    """Функция для создания или обновления векторной базы"""
    logger.info("Начинаем индексацию данных...")
    if not os.path.exists(DATA_PATH):
        return "Ошибка: Папка knowledge_base не найдена!"
    
    # Загрузка
    loader = DirectoryLoader(DATA_PATH, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    # Нарезка
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Сохранение в базу
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    logger.info(f"База обновлена. Создано чанков: {len(chunks)}")
    return f"Успех! База знаний пересобрана. Всего сегментов: {len(chunks)}"

def ask_neostack(question):
    """Функция поиска ответа в базе"""
    # Подключаемся к существующей базе
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Ищем 2 самых похожих куска текста (k=2)
    docs = db.similarity_search(question, k=2)
    
    if not docs:
        return "В базе данных NeoStack пока нет информации по этому запросу."
    
    # Собираем найденный текст в один ответ
    context = "\n\n".join([f"Источник {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
    
    response = f"**NeoStack AI Консультант**\n\nНашел в документах следующие условия:\n\n{context}"
    
    logger.info(f"Пользователь спросил: {question}")
    return response

# --- 3. ИНТЕРФЕЙС ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#NeoStack Bank: Smart RAG Assistant")
    gr.Markdown("Интеллектуальная система поиска по банковским продуктам (Проектная работа).")
    
    with gr.Tab("Чат с поддержкой"):
        msg = gr.Textbox(label="Введите ваш вопрос", placeholder="Например: Какие бонусы за друга?")
        output = gr.Markdown(label="Ответ системы")
        btn = gr.Button("Запросить базу", variant="primary")
        btn.click(ask_neostack, inputs=msg, outputs=output)
    
    with gr.Tab("Управление данными"):
        admin_btn = gr.Button("Обновить векторную базу знаний")
        admin_output = gr.Textbox(label="Статус")
        admin_btn.click(run_indexing, outputs=admin_output)

if __name__ == "__main__":
    # Если базы еще нет — создаем её при запуске
    if not os.path.exists(DB_PATH):
        run_indexing()
    
    # Запуск интерфейса
    demo.launch()