from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate 
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn 
import os
from langchain_community.llms import Ollama

app = FastAPI(
    title = "Langchain Server",
    version="1.0",
    description="A simple API Server"
)

llm1 = Ollama(model = 'mistral')
llm2 = Ollama(model = 'llama3')


prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|llm1,
    path = "/essay"
)

add_routes(
    app,
    prompt2|llm2,
    path = "/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)