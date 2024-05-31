import os
import PyPDF2
import streamlit as st
import pytesseract
import cv2
import request
import json
from PIL import Image
from io import StringIO
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from starlette.requests import Request

st.set_page_config(page_title="Busqueda", page_icon=':shark:')
@st.server_route("/v1/saveImage", methods=["POST"])
def save_text_from_image(req Request):
    input_json = await req.body()
    input_data = json.loads(input_json)
    output_data = input_data

    return json.dumps(output_data)
'''
@st.server_route("/v1/getResponse", methods=["POST"])
def getResponseFromImage():
    input_json = st.server_request.get_json()
    salida2 = 'Salida web service 2'

    return salida2
'''
'''
url = 'www.google.com'
input_data = {
    'param2' : 'valor1'
    'param2' : 'valor2'
}
@st.cache_data
def extract_text_from_image(file_path):
    if os.path.isfile(file_path):
        if file_path.endswith(".jpg"):
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text
@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "BÚSQUEDA DE SIMILITUD":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error al crear el vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)

    return retriever
  '''
    
def main():
    st.write(
        f"""
        <div style="display: flex; align-items: center; margin-left: 0;">
            <h1 style="display: inline-block;">Busqueda</h1>
            <sup style="margin-left:5px;font-size:small; color: blue;">test v0.1</sup>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write ('Prueba de la demo')
'''
    st.sidebar.title("Menú")
    folder_path = "documentos"

    # Nueva opción para seleccionar archivos PDF dentro de la carpeta
    selected_file = st.sidebar.selectbox("Selecciona un archivo:", options=os.listdir(folder_path), index=0)

    embedding_option = st.sidebar.radio(
        "Elige Embeddings", ["OpenAI Embeddings"])

    retriever_type = st.sidebar.selectbox(
        "Elige Retriever", ["BÚSQUEDA DE SIMILITUD"])

    temperature = st.sidebar.slider(
        "Temperatura", 0.0, 1.5, 0.8, step=0.1)
    
    chunk_size = st.sidebar.slider(
        "Tamaño de Chunk (chunk_size)", 100, 2000, 1000, step=100)
    
    splitter_type = "RecursiveCharacterTextSplitter"
    
    start_app = st.sidebar.checkbox("Iniciar", value=False)
    load_files_option = st.sidebar.checkbox("Cargar archivos", value=False)
    load_files_image = st.sidebar.checkbox("Cargar imagen", value=False)


    if start_app:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

    if load_files_image:
        embeddings = HuggingFaceEmbeddings()
        question_answering = pipeline("question-answering")
        file_path = os.path.join(folder_path, selected_file)
        st.write("Ruta:",file_path)
        extracted_text = extract_text_from_image(file_path)
        
        #db = FAISS.from_texts(splits, embeddings)
        st.write("Procesando imagen...")
        
        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            answer = question_answering(question=user_question, context=extracted_text)
            st.write("Respuesta:", answer)

    if load_files_option:
        uploaded_files = st.file_uploader("Sube un documento PDF o TXT", type=[
                                      "pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
                st.session_state.last_uploaded_files = uploaded_files

            loaded_text = load_docs(uploaded_files)
            st.write("Documentos cargados y procesados.")

            splits = split_texts(loaded_text, chunk_size=chunk_size,
                                overlap=0, split_method=splitter_type)

            num_chunks = len(splits)

            if embedding_option == "OpenAI Embeddings":
                embeddings = HuggingFaceEmbeddings()

            retriever = create_retriever(embeddings, splits, retriever_type)

            callback_handler = StreamingStdOutCallbackHandler()
            callback_manager = CallbackManager([callback_handler])

            db = FAISS.from_texts(splits, embeddings)

            user_question = st.text_input("Ingresa tu pregunta:")
            if user_question:
                answer = db.similarity_search(user_question)
                st.write("Respuesta:", answer)
    else:
        file_path = os.path.join(folder_path, selected_file)
        loaded_text = load_Documentos(file_path)
        splits = split_texts(loaded_text, chunk_size=chunk_size,
                             overlap=0, split_method=splitter_type)

        num_chunks = len(splits)
        st.write(f"Número de chunks: {num_chunks}")

        if embedding_option == "OpenAI Embeddings":
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)

        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

     
        db = FAISS.from_texts(splits, embeddings)
        

        st.write("Listo para responder preguntas.")

        user_question = st.text_input("Ingresa tu pregunta:")
        if user_question:
            answer = db.similarity_search(user_question)
            st.write("Respuesta:", answer)
'''  

if __name__ == "__main__":
    main()
