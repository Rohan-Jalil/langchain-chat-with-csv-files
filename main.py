import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DocArrayInMemorySearch

load_dotenv()

loader = CSVLoader(
    file_path=os.path.join("files_to_read", "test.csv"), encoding="utf-8"
)
data = loader.load()

embeddings = OpenAIEmbeddings()
vector_store = DocArrayInMemorySearch.from_documents(data, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)

question = "How many totals rows are there in the dataset?"
result = qa({"query": question})

print(result)
