import json
import os
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


DEFAULT_DATASET_PATH = "../../dataset/2016"
DEFAULT_OUTPUT_FILE = "cache/cpc_top10_2016_mapping.json"
DEFAULT_TOP_N = 10


class CPCAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.cpc_to_files: Dict[str, List[str]] = defaultdict(list)
        self.cpc_counts: Dict[str, int] = defaultdict(int)
    
    def analyze(self):
        json_files = list(self.dataset_path.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                main_cpc = data.get("main_cpc_label", "")
                if main_cpc:
                    self.cpc_to_files[main_cpc].append(file_path.name)
                    self.cpc_counts[main_cpc] += 1
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
    
    def get_top_n(self, n: int = 10) -> List[Tuple[str, int]]:
        return sorted(self.cpc_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def select_random_patents(self, cpc_code: str, num_patents: int = 10) -> List[str]:
        files = self.cpc_to_files.get(cpc_code, [])
        if len(files) <= num_patents:
            return files
        return random.sample(files, num_patents)
    
    def save_results(self, output_path, top_n = 10, random_sample_size = 10):
        top_cpc = self.get_top_n(top_n)
        
        results = {
            "top_subclasses": [
                {
                    "cpc_code": cpc,
                    "count": count,
                    "files": self.cpc_to_files[cpc],
                    "random_files": self.select_random_patents(cpc, random_sample_size)
                }
                for cpc, count in top_cpc
            ],
            "summary": {
                "total_unique_cpc_codes": len(self.cpc_counts),
                "total_patents_analyzed": sum(self.cpc_counts.values()),
                "top_n": top_n
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        print(f"\nTop {top_n} CPC Subclasses:")
        for cpc, count in top_cpc:
            print(f"  {cpc}: {count} patents")


def main():
    analyzer = CPCAnalyzer(DEFAULT_DATASET_PATH)
    print(f"Analyzing patents in {DEFAULT_DATASET_PATH}...")
    analyzer.analyze()
    analyzer.save_results(DEFAULT_OUTPUT_FILE, DEFAULT_TOP_N)


if __name__ == "__main__":
    main()
