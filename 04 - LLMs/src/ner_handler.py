import warnings, re
warnings.filterwarnings(
    "ignore",
    message=re.escape("CUDA initialization: CUDA unknown error"),
    category=UserWarning
)
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class NERHandler:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model_name = "sm"
        hardware = os.getenv("HARDWARE", "CPU").upper()
        if hardware == "GPU":
            try:
                spacy.prefer_gpu()
                self.device = "gpu"
                self.nlp = spacy.load("en_core_web_trf")
                self.model_name = "trf"
                spacy.prefer_gpu()
                print("NERHandler using GPU")
            except Exception as e:
                print(f"GPU not available for spacy, falling back to CPU: {e}")
                self.device = "cpu"
        else:
            self.device = "cpu"
            print("NERHandler using CPU")
    
    def get_model_name(self):
        return self.model_name

    def extract_from_text(self, abstract, max_words=3):
        doc = self.nlp(abstract)
        candidates = set()
        for np in doc.noun_chunks:
            phrase = np.text.strip()
            if 1 <= len(phrase.split()) <= max_words:
                candidates.add(phrase)
        return list(candidates)
    
    def clean_entities(self, entities):
        cleaned = []
        for ent in entities:
            text = ent.strip().lower()
            words = [w for w in text.split() if w not in STOP_WORDS]

            if not words:
                continue

            phrase = " ".join(words).strip()

            if len(phrase) < 3:
                continue

            blacklist = {
                "method", "apparatus", "system", "device", "composition", 
                "components", "process", "regimen", "embodiments", "invention"
            }

            if phrase in blacklist:
                continue

            cleaned.append(phrase)

        final = list(dict.fromkeys(cleaned))
        return final


    def extract_entities(self, text):
        raw_entities = self.extract_from_text(text)
        entities = self.clean_entities(raw_entities)
        return entities