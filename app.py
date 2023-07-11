import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch

load_dotenv()

uploaded_file = st.file_uploader("Choose a .csv file", "csv")
question = st.text_input("Enter your query:")

if st.button("Submit"):
    if uploaded_file is not None and question != "":
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()

        embeddings = OpenAIEmbeddings()
        vector_store = DocArrayInMemorySearch.from_documents(data, embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )

        result = qa({"query": question})

        st.write(result["result"])
    else:
        st.error("Please upload a document and enter a query!")
