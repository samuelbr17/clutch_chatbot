from src.context_filter_llm_model import context_filter_llm_model
from src.sql_llm_model import sql_llm_model
import pandas as pd
from gpt4all import GPT4All
from langchain.llms import GPT4All


class chatbot_llm_model():
    def __init__(self, 
                 database_path: str, 
                 questions_df_path: str, 
                 chatbot_model:  str = "ggml-model-gpt4all-falcon-q4_0.bin",
                 context_filter_model: str = 'bert-base-nli-mean-tokens',
                 sql_model: str = "ggml-model-gpt4all-falcon-q4_0.bin"):
        self.model = GPT4All(model=chatbot_model)
        self.context_filter_llm_model = context_filter_llm_model(questions_df=pd.read_csv(questions_df_path))
        self.sql_llm_model = sql_llm_model(database_path = database_path)

    def answer_query(self, query: str):
        
        table = 'clients'
        columns = ['Status', 'ID',  'ModificationDate', 'Rate', 'Amount']

        data_answer = self.sql_llm_model.get_question_reponse(table, query, columns)

        context = f"""
        You are a friendly loan assistant from a lending startup that responds in a conversational
        manner to users questions. Keep the answers short, unless specifically
        asked by the user to elaborate on something. Always maintain a polite and positive sentiment tone.

        You need to politily answer the question: {query}

        The numeric answer is: {data_answer}, develop an answer in the chat

        """

        print(self.model(context))
        

    def execute_query(self, query: str):
        context_score = self.context_filter_llm_model.get_context_score(query=query)

        if context_score == -1:
            return ("Sorry, I cannot answer that question") 
        elif context_score == 0:
            return ("I appreciate your question, but it's quite complex and may require the expertise of a more senior lending officer. They can provide you with a more detailed and accurate response. Please contact them directly for assistance with your inquiry.")
        else:
            return self.answer_query(query)
        