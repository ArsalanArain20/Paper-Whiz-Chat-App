import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def PDF_DOC(pdf):
    text = ""
    for pdf in pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def PDF_Chunck(doc):
    doc_splitting = RecursiveCharacterTextSplitter(chunk_size=15000,chunk_overlap=1200)
    chunks = doc_splitting.split_text(doc)
    return chunks

def Vector_Stored(doc_chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vec_store = FAISS.from_texts(doc_chunk,embedding=embeddings)
    vec_store.save_local("faiss_index")

def Conversional_Chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3, explain=True)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = Conversional_Chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Paper Whiz Chat")
    st.header("Paper Whiz Chat 👨‍💻")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = PDF_DOC(pdf_docs)
                text_chunks = PDF_Chunck(raw_text)
                Vector_Stored(text_chunks)
                st.success("Done")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
