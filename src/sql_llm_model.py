from gpt4all import GPT4All
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import psycopg2
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain.callbacks import get_openai_callback
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine, MetaData
from llama_index import LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from langchain import OpenAI
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os


class sql_llm_model:
    def __init__(
        self, database_path: str, model: str = "ggml-model-gpt4all-falcon-q4_0.bin"
    ):
        self.model = GPT4All(model=model)
        self.database = SQLDatabase.from_uri(f"sqlite:///{database_path}")

        template = "Write a SQL Query given the table named {table} and columns as a list {columns} for the given question : {question}. All variables are independent, no need to call all of them in the query."
        self.prompt = PromptTemplate(
            template=template, input_variables=["table", "question", "columns"]
        )

        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.model)

    def get_sql_query(self, table: str, question: str, columns: (list | str)) -> str:
        llm_chain = self.llm_chain

        response = llm_chain.run(
            {"table": table, "question": question, "columns": columns}
        )
        return response

    def get_question_reponse(self, table: str, question: str, columns: (list | str)):
        query = self.get_sql_query(table, question, columns)
        return self.database.run(query)


if __name__ == "__main__":
    sql_llm_model = sql_llm_model()

    table = "clients"
    columns = ["Status", "ID", "ModificationDate", "Rate", "Amount"]
    question = "What is the amount from client 1?"

    print(sql_llm_model.get_llm_response(table, question, columns))
