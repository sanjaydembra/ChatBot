import os
import torch
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

vectorstore = None
conversation_chain = None
chat_history = []

# Extract text from uploaded PDFs
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks for vector storage
def split_text_into_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

# Create a FAISS vectorstore with HuggingFace embeddings
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Initialize the LLM-based conversation chain
def initialize_conversation_chain(vstore):
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.5,
        do_sample=False,
        device=-1  # -1 = CPU
    )

    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent academic assistant. Answer the question based only on the context provided.
Be concise, clear, and avoid copying large chunks of context. If not found, say you don’t know.

Context:
{context}

Question: {question}

Answer:
"""
    )

    retriever = vstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global vectorstore, conversation_chain
    uploaded_pdfs = request.files.getlist('pdf_docs')

    raw_text = extract_pdf_text(uploaded_pdfs)
    chunks = split_text_into_chunks(raw_text)
    vectorstore = create_vectorstore(chunks)
    conversation_chain = initialize_conversation_chain(vectorstore)

    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversation_chain, chat_history

    if not conversation_chain:
        return redirect('/')

    if request.method == 'POST':
        user_input = request.form['user_question']
        response = conversation_chain({'question': user_input})
        chat_history = [(msg.content, 'human' if msg.type == 'human' else 'ai') for msg in response['chat_history']]

    return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)