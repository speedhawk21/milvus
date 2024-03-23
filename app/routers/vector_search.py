from dotenv import load_dotenv
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus

from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()


@router.get("/")
def search_vector():
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

    vector_store = Milvus(
        connection_args={"host": "localhost", "port": 19530},
        collection_name="millertime",
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
        'job_title': "VP of Marketing",
        'is_decision_maker': "Yes",
        'stage_of_sales': "Qualification",
        'challenges': "Competitor has a better product",
        'deal_risk': "High",
        'deal_size': "Large",
        'custom_notes': "Customer is interested in a long term partnership",
        'question': "What is the best introduction to a VP of Marketing at coca cola for our Tableau Server product?",
        'solution_features': "Advanced data visualization, real-time analytics",
        'key_benefits': "Enhanced decision-making capabilities, increased ROI",
        'customization_options': "Custom dashboards, branded reports",
        'integration_points': "CRM integration, social media analytics"
    }

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(data)
