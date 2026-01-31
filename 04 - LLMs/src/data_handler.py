import json
import os
from pathlib import Path
from ner_handler import NERHandler
from tqdm import tqdm

class DataHandler:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.ner = NERHandler()
    
    def get_ner_model_name(self):
        return self.ner.get_model_name()
    
    def _load_texts_and_create_cache(self, dataset_tag=None, ner_model_name=None, data_file=None):
        docs = []
        texts = []

        files = list(self.folder.glob("*.json"))
        for file in tqdm(files, desc="Processing patent files", leave=True):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "")
            abstract = data.get("abstract", "")
        
            text = "\n".join([
                title, abstract
            ]).strip()

            metadata = {
                "filing_date": data.get("filing_date"),
                "patent_issue_date": data.get("patent_issue_date"),
            }

            entities = self.ner.extract_entities(abstract)
            text = f"{text}\nEntities: {entities}"

            docs.append({  
                "id": file.stem,
                "text": text, 
                "entities": entities,    
                "metadata": metadata, 
            })

            texts.append(text)

        if dataset_tag and ner_model_name:
            data_file = f"cache/data_{dataset_tag}_{ner_model_name}.json"
            with open(data_file, "w") as f:
                json.dump(docs, f, indent=2)
            print(f"Cache saved to {data_file}")
        
        return texts, docs
        
    def load_texts(self, dataset_tag=None, ner_model_name=None):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.join(BASE_DIR, "cache")
        data_file = os.path.join(CACHE_DIR, f"data_{dataset_tag}_{ner_model_name}.json")
        
        if dataset_tag and ner_model_name:
            try:
                with open(data_file, "r") as f:
                    docs = json.load(f)
                    texts = [doc["text"] for doc in docs]
                    print(f"Loaded {len(docs)} documents from cache: {data_file}")
                    return texts, docs

            except FileNotFoundError:
                print(f"Cache file {data_file} not found, processing files...")
        
        texts, docs = self._load_texts_and_create_cache(dataset_tag, ner_model_name, data_file)
        return texts, docs
        