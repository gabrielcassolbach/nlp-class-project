class PatentsRAG:
    def __init__(self, faiss_index, vectorizer, score_threshold=0.7):
        self.index = faiss_index
        self.vectorizer = vectorizer
        self.score_threshold = score_threshold

    def truncate_context(self, results, chars_per_result):
        truncated = [text[:chars_per_result] for text in results]
        return "\n".join(truncated)

    def generate_prompt(self, query_text, top_k, model_name="all-MiniLM-L6-v2"):
        if model_name == "BAAI/bge-large-en-v1.5":
            query_search = "Represent this sentence for searching relevant passages: " + query_text
            query_vec = self.vectorizer.encode_texts([query_search])[0]
        else:
            query_vec = self.vectorizer.encode_texts([query_text])[0]
        filtered_docs = self.index.retrieve(query_vec, top_k=top_k)
        
        filtered_docs = [doc for doc in filtered_docs if doc["score"] >= self.score_threshold]

        print(filtered_docs)

        if not filtered_docs:
            return "insufficient evidence"
        
        context_blocks = []
        
        for doc in filtered_docs:
            score_pct = f"{doc['score'] * 100:.2f}%"
            entities = ", ".join(doc["entities"])
            excerpt = doc.get("text")

            if len(excerpt) > 1500:
                excerpt = excerpt[:1500] + " ...[truncated]"
            block = (
                f"PATENT {doc['id']} (score: {score_pct})\n"
                f"Excerpt:\n{excerpt}\n"
                f"Entities:\n{entities}\n"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)
        patent_ids = ", ".join([doc["id"] for doc in filtered_docs])
        patent_scores = ", ".join(
            [f"{doc['score'] * 100:.2f}% precision" for doc in filtered_docs]
        )

        for doc in filtered_docs: 
            print(doc["score"])
        
        prompt = f"""
        You are an expert patent analyst. Follow these strict rules:

        1) Use ONLY the information explicitly present in the CONTEXT.  
        Do NOT add external knowledge, interpretations, generalizations, or assumptions.

        2) If the answer cannot be fully supported by the CONTEXT, respond exactly:
        "insufficient evidence"

        3) Every factual statement must cite the supporting patent ID(s) in parentheses.

        4) At the end of the output, append exactly:
        Referenced patent(s): {patent_ids}
        Patent Score(s): {patent_scores}

        QUESTION:
        {query_text}

        CONTEXT:
        {context_text}

        """

        return prompt, [f"{doc['id']}.json" for doc in filtered_docs]