import numpy as np
import sys
import os
import pickle

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../04 - LLMs/src"))
sys.path.append(module_path)

from pathlib import Path
from vectorizer import Vectorizer
from faiss_index import FAISSIndex
from rag import PatentsRAG
from data_handler import DataHandler
from google import genai
from llm import LLM
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[3]  
DATA_PATH = ROOT / "data"

class RAGService:
    def __init__(self, USE_SAMPLE):
        load_dotenv() 
        # self.model_name = "BAAI/bge-large-en-v1.5"
        self.model_name = "all-MiniLM-L6-v2"
        self.vectorizer = Vectorizer(model_name=self.model_name)
        dataset_tag = "sample" if USE_SAMPLE else "2016"
        self.data_handler = DataHandler(DATA_PATH)

        texts, docs = self.data_handler.load_texts(
            dataset_tag=dataset_tag,
            ner_model_name=self.data_handler.get_ner_model_name()
        )

        self.vectors, self.docs = self.vectorizer.compute_vectors_and_metadata(
            texts, docs, dataset_tag,
            batch_size=64
        )

        self.faiss_index = FAISSIndex(self.vectors, docs)
        self.rag = PatentsRAG(self.faiss_index, self.vectorizer, score_threshold=0.1)
        self.llm = LLM(client=genai.Client(api_key= os.getenv("API_KEY")), model="gemini-2.5-flash")

    def ask(self, query: str, top_k: int = 3) -> str:
        prompt, _ = self.rag.generate_prompt(query, top_k, self.model_name)
        return self.llm.generate_text(prompt)