import os
from dotenv import load_dotenv
from patent_rag_system import PatentRAGSystem

load_dotenv()

def main():
    SAMPLE_DATASET_PATH = os.getenv("SAMPLE_DATASET_PATH")
    TOTAL_DATASET_PATH = os.getenv("TOTAL_DATASET_PATH")
    
    USE_SAMPLE = True  # False to use TOTAL_DATASET_PATH
    dataset_path = SAMPLE_DATASET_PATH if USE_SAMPLE else TOTAL_DATASET_PATH
    dataset_tag = "sample" if USE_SAMPLE else "2016"
    
    model_name = "BAAI/bge-large-en-v1.5"
    # model_name = "all-MiniLM-L6-v2"
    
    print(f"Initializing Patent RAG System with model: {model_name}...")
    rag_system = PatentRAGSystem(
        dataset_path=dataset_path,
        model_name=model_name,
        dataset_tag=dataset_tag,
        score_threshold=0.1
    )
    
    print("Enter your question: ")
    query = input()
    
    answer, retrieved_files, scores = rag_system.generate_answer(query, top_k=3)
    
    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== TOP 3 SIMILAR PATENTS ===")
    for i, (filename, score) in enumerate(zip(retrieved_files, scores), 1):
        print(f"{i}. {filename} (score: {score:.4f})")

if __name__ == "__main__":
    main()

## frontend com esse backend.
## testar outro embedding e comparar ele
## se aprofundar na identificação de palavras chaves -> melhorar a técnica.
