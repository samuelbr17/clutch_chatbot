
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class context_filter_llm_model():
    def __init__(self, 
                 questions_df: pd.DataFrame, 
                 model: str = 'bert-base-nli-mean-tokens'):
        self.questions_df = questions_df
        self.model = SentenceTransformer(model)
        self.similarity_threshold = 0.75
        self.boolean_threshold = 0.60

    def get_similarities(self, query: str) -> np.ndarray:

        sentences = self.questions_df['Example Question']
        
        sentences_embeddings = self.model.encode(sentences)
        query_embedding = self.model.encode(query)

        similarities = cosine_similarity(
                                        [query_embedding],
                                        sentences_embeddings)[0]

        return similarities
    
    def get_most_similar_questions(self, query:str) -> pd.DataFrame:
        df = self.questions_df
        df['similarity'] = self.get_similarities(query)

        # filter by similarity threshold
        df = df.loc[(df.similarity > self.similarity_threshold)]
        # select all the max similarities
        df = df.loc[df.similarity == df.similarity.max()]

        return df
    
    
    def filter_context(self, query: str):

        df = self.get_most_similar_questions(query)

        # take the mean of similarities (1= True, 0= False)
        boolean_score = df['Hal-Answers'].mean()

        return boolean_score > self.boolean_threshold
