class LLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate_content(self, prompt):
        return self.client.models.generate_content(model=self.model, contents=prompt)

    def generate_text(self, prompt):
        response = self.generate_content(prompt)
        return response.candidates[0].content.parts[0].text