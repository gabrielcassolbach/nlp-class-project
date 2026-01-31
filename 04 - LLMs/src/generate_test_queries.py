import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from llm import LLM

load_dotenv()

DEFAULT_MAPPING_FILE = "cache/cpc_top10_2016_mapping.json"
DEFAULT_DATASET_PATH = "../../dataset/2016"
DEFAULT_OUTPUT_FILE = "cache/generated_queries.json"
WAIT_TIME = 90  # 1 minute and 30 seconds


def load_mapping(mapping_file: str):
    """Load the CPC mapping file with random samples."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_patent_description(dataset_path: str, filename: str):
    file_path = Path(dataset_path) / filename
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("full_description", "")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filename}: {e}")
        return None


def generate_query_for_patent(llm: LLM, full_description: str):
    """Generate a one-sentence query from the full description."""
    if not full_description:
        return None
    
    # Truncate to avoid token limits
    truncated_desc = full_description[:3000]
    
    prompt = f"""Read the following patent description and generate a single, specific search query (one sentence) that a researcher would use to find this technology. 
    Do not mention the patent number or make it obvious you're describing this exact patent.
    Focus on the key technical innovation or problem being solved. Make your query like it should in a search engine, with few words.

    Patent Description:
    {truncated_desc}

        Generate only the query sentence, nothing else:"""
    
    try:
        return llm.generate_text(prompt)
    except Exception as e:
        print(f"Error generating query: {e}")
        return None


def main():
    client = genai.Client(api_key=os.getenv("API_KEY"))
    llm = LLM(client=client, model="gemini-2.0-flash")
    
    mapping = load_mapping(DEFAULT_MAPPING_FILE)
    
    results = []
    total_queries = 0
    
    # Loop through top 10 CPC codes
    for idx, subclass in enumerate(mapping["top_subclasses"]):
        cpc_code = subclass["cpc_code"]
        random_files = subclass.get("random_files", [])
        
        print(f"\n{'='*60}")
        print(f"Processing CPC Code: {cpc_code} ({idx+1}/10)")
        print(f"{'='*60}")
        
        cpc_results = {
            "cpc_code": cpc_code,
            "queries": []
        }
        
        for file_idx, filename in enumerate(random_files):
            print(f"\n  [{file_idx+1}/{len(random_files)}] Processing: {filename}")
            
            full_description = load_patent_description(DEFAULT_DATASET_PATH, filename)
            
            if full_description is None:
                print(f"    Skipping due to error...")
                cpc_results["queries"].append({
                    "filename": filename,
                    "query": None,
                    "error": "Could not load patent"
                })
                continue
            
            query = generate_query_for_patent(llm, full_description)
            
            if query:
                print(f"    Generated query: {query[:100]}...")
                cpc_results["queries"].append({
                    "filename": filename,
                    "query": query
                })
            else:
                print(f"    Failed to generate query")
                cpc_results["queries"].append({
                    "filename": filename,
                    "query": None,
                    "error": "Query generation failed"
                })
            
            total_queries += 1
        
        results.append(cpc_results)
        
        # Wait after processing every 10 patents, given API rate limits
        if idx < len(mapping["top_subclasses"]) - 1:
            print(f"\n  Waiting {WAIT_TIME} seconds before next CPC code...")
            time.sleep(WAIT_TIME)
    
    with open(DEFAULT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Query generation complete!")
    print(f"Total queries generated: {total_queries}")
    print(f"Results saved to: {DEFAULT_OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
