from fastapi import FastAPI, HTTPException
from .routers import vector_setup, vector_search

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

os.environ["OCTOAI_API_TOKEN"] = os.getenv("OCTOAI_API_TOKEN", "default_value")

app = FastAPI()

app.include_router(vector_setup.router, prefix="/vectorsetup")
app.include_router(vector_search.router, prefix="/vector_search")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}
