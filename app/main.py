import os
from fastapi import FastAPI
from .routers import vector_setup, vector_search

os.environ["OCTOAI_API_TOKEN"] = os.getenv("OCTOAI_API_TOKEN", "default_value")

app = FastAPI()

app.include_router(vector_setup.router, prefix="/vectorsetup")
app.include_router(vector_search.router, prefix="/vector_search")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"Hello": "World"}
