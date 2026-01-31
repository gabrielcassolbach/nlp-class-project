import os
from dotenv import load_dotenv
from data_handler import DataHandler
from vectorizer import Vectorizer
from faiss_index import FAISSIndex
from rag import PatentsRAG
from google import genai
from llm import LLM

load_dotenv()

class PatentRAGSystem:
    
    def __init__(self, dataset_path, model_name = "all-MiniLM-L6-v2", dataset_tag = "sample", score_threshold = 0.1, batch_size = 64):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.dataset_tag = dataset_tag
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        
        # Initialize components
        self.data_handler = DataHandler(dataset_path)
        self.ner_model_name = self.data_handler.get_ner_model_name()
        
        # Load and process data
        texts, docs = self.data_handler.load_texts(
            dataset_tag=dataset_tag,
            ner_model_name=self.ner_model_name
        )
        
        # Compute or load cached vectors
        self.vectorizer = Vectorizer(model_name=model_name)
        self.vectors, self.docs = self.vectorizer.compute_vectors_and_metadata(
            texts, docs, dataset_tag,
            batch_size=batch_size
        )
        
        # Create FAISS index and RAG
        self.faiss_index = FAISSIndex(self.vectors, self.docs)
        self.rag = PatentsRAG(self.faiss_index, self.vectorizer, score_threshold=score_threshold)
    
    def query(self, query_text, top_k = 3):
        prompt, retrieved_files = self.rag.generate_prompt(query_text, top_k=top_k)
        scores = self._get_scores(query_text, retrieved_files)
        
        return prompt, retrieved_files, scores
    
    def _get_scores(self, query_text, retrieved_files):
        """Get similarity scores for retrieved files."""
        results = self.faiss_index.retrieve(
            self.vectorizer.model.encode([query_text], convert_to_numpy=True),
            top_k=len(retrieved_files)
        )
        return [result["score"] for result in results]
    
    def generate_answer(self, query_text, top_k = 3, llm_model = "gemini-2.5-flash"):
        prompt, retrieved_files, scores = self.query(query_text, top_k=top_k)
        
        llm = LLM(
            client=genai.Client(api_key=os.getenv("API_KEY")),
            model=llm_model
        )
        answer = llm.generate_text(prompt)
        
        return answer, retrieved_files, scores
