from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus

from fastapi import APIRouter
from typing import List

router = APIRouter()


@router.get("/")
def search_vector(job_title: str, is_decision_maker: bool, stage_of_sales: str, challenges: str, deal_risk: str, deal_size: str, custom_notes: str, question: str, solution_features: str, key_benefits: str, customization_options: str, integration_points: str):
    llm = OctoAIEndpoint(
        endpoint_url="https://text.octoai.run/v1/chat/completions",
        model_kwargs={
            "model": "mixtral-8x7b-instruct-fp16",
            "max_tokens": 1024,
            "presence_penalty": 0,
            "temperature": 0.01,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
                },
            ],
        },
    )

    embeddings = OctoAIEmbeddings(
        endpoint_url="https://text.octoai.run/v1/embeddings")

    vector_store = Milvus(
        connection_args={"host": "localhost", "port": 19530},
        collection_name="millertime",
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    You have knowledge around Miller Heiman sales methodologies and overall business development strategies. Below is an instruction that describes a request from a sales agent looking for helpful information to provide a customer or help close a deal.
    The target customer that the agent is speaking with has a job title of {job_title}. Apply the 6 core concepts of solution selling that can benefit a sales agent with these additional details:
    - Decision Maker: {is_decision_maker}
    - Sales Stage: {stage_of_sales}
    - Main Challenges: {challenges}
    - Deal Risk: {deal_risk}
    - Deal Size: {deal_size}
    - Custom Notes: {custom_notes}
    - Potential Solution Features: {solution_features}
    - Key Benefits: {key_benefits}
    - Customization Options: {customization_options}
    - Integration Points: {integration_points}
    Write a response that appropriately completes the request.
    Instruction:
    {question}
    Response: """

    prompt = PromptTemplate.from_template(template)

    data = {
        'context': retriever,
        'job_title': job_title,
        'is_decision_maker': is_decision_maker,
        'stage_of_sales': stage_of_sales,
        'challenges': challenges,
        'deal_risk': deal_risk,
        'deal_size': deal_size,
        'custom_notes': custom_notes,
        'question': question,
        'solution_features': solution_features,
        'key_benefits': key_benefits,
        'customization_options': customization_options,
        'integration_points': integration_points,
    }

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(data)

    return response
