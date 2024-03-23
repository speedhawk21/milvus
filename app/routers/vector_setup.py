
from dotenv import load_dotenv
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus

from fastapi import APIRouter, HTTPException
from typing import List

import pandas as pd

router = APIRouter()


@router.get("/")
def create_vector():
    try:
        embeddings = OctoAIEmbeddings(
            endpoint_url="https://text.octoai.run/v1/embeddings")

        pdf_file = "millertime/data/TheNewStrategicSelling.pdf"
        pdf_loader = PyPDFLoader(pdf_file)
        data = pdf_loader.load()
        data_df = pd.DataFrame(data)

        # Iterate through dataframe and create a document for each row
        documents = []
        for i, row in data_df.iterrows():
            document = Document(
                page_content=row["page_content"],
                metadata={"page": row["metadata"]["page"]}
            )
            documents.append(document)

        vector_store = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            connection_args={"host": "localhost", "port": 19530},
            collection_name="millertime"
        )

        return {"message": "Vector created successfully"}
    except Exception as e:
        return {"error": str(e)}
