import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Sidebar content
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LLAMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) LLM model
    ''')
    add_vertical_space(5)
    st.write('Created by Tushar Kumar')

def main():
    st.header("ASK YOUR DOCUMENT🤗💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text:
            st.error("The uploaded PDF file is empty or not readable.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from Disk')
        else:
            embeddings = OllamaEmbeddings(model='llama3')
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        prompt_template = """
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        Question: {input}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        llm = Ollama(model='llama3')
        document_chain = create_stuff_documents_chain(llm, prompt, output_parser=StrOutputParser())
        retriever = VectorStore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        input_text = st.text_input("Search the topic you want")
        if input_text:
            response = retrieval_chain.invoke({"input": input_text})
            if 'answer' in response:
                st.write(response['answer'])
            else:
                st.error("An error occurred during processing. Please try again.")

if __name__ == '__main__':
    main()
