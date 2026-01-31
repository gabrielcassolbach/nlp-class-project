import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from patent_rag_system import PatentRAGSystem

load_dotenv()


class BenchmarkSystem:

    def __init__(self, dataset_path, dataset_tag, queries_file):

        self.dataset_path = dataset_path
        self.dataset_tag = dataset_tag
        self.queries = self._load_queries(queries_file)
        print(f"Loaded {len(self.queries)} test queries from {queries_file}")
    
    def _load_queries(self, queries_file):

        with open(queries_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = []
        for cpc_entry in data:
            cpc_code = cpc_entry["cpc_code"]
            for query_item in cpc_entry["queries"]:
                
                if query_item.get("query") is None:
                    continue
                
                queries.append({
                    "cpc_code": cpc_code,
                    "query": query_item["query"].strip(),  # Remove trailing \n
                    "ground_truth_file": query_item["filename"]
                })
        
        return queries
    
    def evaluate_model(self, model_name, top_k=5):
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        # Initialize RAG system with the specified model
        rag_system = PatentRAGSystem(
            dataset_path=self.dataset_path,
            model_name=model_name,
            dataset_tag=self.dataset_tag,
            score_threshold=0.1
        )
        
        results = []
        
        # Iterate through all queries with progress bar
        for query_item in tqdm(self.queries, desc=f"Processing queries", leave=True):
            query_text = query_item["query"]
            ground_truth = query_item["ground_truth_file"]
            cpc_code = query_item["cpc_code"]
            
            try:
                # Retrieve top-k documents
                _, retrieved_files, scores = rag_system.query(query_text, top_k=top_k)
                
                # Check hits at different K values
                hit_at_1 = ground_truth in retrieved_files[:1] if len(retrieved_files) >= 1 else False
                hit_at_3 = ground_truth in retrieved_files[:3] if len(retrieved_files) >= 3 else False
                hit_at_5 = ground_truth in retrieved_files[:5] if len(retrieved_files) >= 5 else False
                
                # Calculate reciprocal rank for MRR
                reciprocal_rank = 0.0
                if ground_truth in retrieved_files:
                    rank = retrieved_files.index(ground_truth) + 1  # 1-indexed
                    reciprocal_rank = 1.0 / rank
                
                results.append({
                    "cpc_code": cpc_code,
                    "query": query_text,
                    "ground_truth_file": ground_truth,
                    "retrieved_files": retrieved_files,
                    "scores": scores,
                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "hit_at_5": hit_at_5,
                    "reciprocal_rank": reciprocal_rank
                })
                
            except Exception as e:
                print(f"\nError processing query '{query_text[:50]}...': {e}")
                results.append({
                    "cpc_code": cpc_code,
                    "query": query_text,
                    "ground_truth_file": ground_truth,
                    "retrieved_files": [],
                    "scores": [],
                    "hit_at_1": False,
                    "hit_at_3": False,
                    "hit_at_5": False,
                    "reciprocal_rank": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def calculate_metrics(self, results):
        
        total_queries = len(results)
        
        if total_queries == 0:
            return {
                "hit_rate_at_1": 0.0,
                "hit_rate_at_3": 0.0,
                "hit_rate_at_5": 0.0,
                "mrr": 0.0,
                "total_queries": 0
            }
        
        hit_count_at_1 = sum(1 for r in results if r["hit_at_1"])
        hit_count_at_3 = sum(1 for r in results if r["hit_at_3"])
        hit_count_at_5 = sum(1 for r in results if r["hit_at_5"])
        
        total_reciprocal_rank = sum(r["reciprocal_rank"] for r in results)
        
        return {
            "hit_rate_at_1": hit_count_at_1 / total_queries,
            "hit_rate_at_3": hit_count_at_3 / total_queries,
            "hit_rate_at_5": hit_count_at_5 / total_queries,
            "mrr": total_reciprocal_rank / total_queries,
            "total_queries": total_queries
        }
    
    def save_results(self, results, output_file):

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
    
    def print_comparison_table(self, metrics_bge: Dict, metrics_allMini: Dict):
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS COMPARISON")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'BGE-Large':<20} {'AllMini':<20}")
        print(f"{'-'*60}")
        print(f"{'Hit Rate @ 1':<20} {metrics_bge['hit_rate_at_1']:<20.4f} {metrics_allMini['hit_rate_at_1']:<20.4f}")
        print(f"{'Hit Rate @ 3':<20} {metrics_bge['hit_rate_at_3']:<20.4f} {metrics_allMini['hit_rate_at_3']:<20.4f}")
        print(f"{'Hit Rate @ 5':<20} {metrics_bge['hit_rate_at_5']:<20.4f} {metrics_allMini['hit_rate_at_5']:<20.4f}")
        print(f"{'MRR':<20} {metrics_bge['mrr']:<20.4f} {metrics_allMini['mrr']:<20.4f}")
        print(f"{'Total Queries':<20} {metrics_bge['total_queries']:<20} {metrics_allMini['total_queries']:<20}")
        print(f"{'='*60}\n")


def main():

    TOTAL_DATASET_PATH = os.getenv("TOTAL_DATASET_PATH")
    QUERIES_FILE = "cache/generated_queries.json"
    OUTPUT_FILE = "cache/benchmark_results.json"
    
    MODEL_BGE_LARGE = "BAAI/bge-large-en-v1.5"
    MODEL_ALLMINI = "all-MiniLM-L6-v2"
    
    benchmark = BenchmarkSystem(
        dataset_path=TOTAL_DATASET_PATH,
        dataset_tag="2016",
        queries_file=QUERIES_FILE
    )
    
    results_bge = benchmark.evaluate_model(MODEL_BGE_LARGE, top_k=5)
    metrics_bge = benchmark.calculate_metrics(results_bge)
    
    results_allMini = benchmark.evaluate_model(MODEL_ALLMINI, top_k=5)
    metrics_allMini = benchmark.calculate_metrics(results_allMini)
    
    benchmark.print_comparison_table(metrics_bge, metrics_allMini)
    
    full_results = {
        "benchmark_config": {
            "dataset_path": TOTAL_DATASET_PATH,
            "dataset_tag": "2016",
            "queries_file": QUERIES_FILE,
            "top_k": 5,
            "score_threshold": 0.1
        },
        "models": {
            "bge_large": {
                "model_name": MODEL_BGE_LARGE,
                "metrics": metrics_bge,
                "detailed_results": results_bge
            },
            "allMini": {
                "model_name": MODEL_ALLMINI,
                "metrics": metrics_allMini,
                "detailed_results": results_allMini
            }
        }
    }
    
    benchmark.save_results(full_results, OUTPUT_FILE)
    
    print("Benchmark evaluation complete!")


if __name__ == "__main__":
    main()
