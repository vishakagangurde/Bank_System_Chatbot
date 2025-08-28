import os
import pandas as pd
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity



class CustomerSupportAgent:
    def __init__(self, data_path, index_path, embed_cache_path):
        self.data_path = data_path
        self.index_path = index_path
        self.embed_cache_path = embed_cache_path
        self.df = pd.read_csv(data_path)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load embeddings cache
        with open(embed_cache_path, "rb") as f:
            self.embedding_cache = pickle.load(f)

        # Gemini model
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def get_embedding(self, text):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        else:
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )["embedding"]
            self.embedding_cache[text] = emb
            return emb

    def generate_answer(self, query, context):
        prompt = f"""You are a helpful banking customer support agent. 
        Use the context below to answer the question accurately.
        If you are unsure, say you don't know.

        Context:
        {context}

        Question: {query}

        Respond in JSON format with keys:
        - response: the actual answer
        - confidence: "low", "medium", or "high"
        """
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            import json
            response_json = json.loads(response_text)
            return response_json
        except Exception as e:
            return {"response": "Error generating answer.", "confidence": "low"}

    def ask_question(self, question):
        # Embed query
        query_vec = np.array(self.get_embedding(question), dtype=np.float32).reshape(1, -1)

        # Retrieve top 3 similar responses
        _, indices = self.index.search(query_vec, k=3)
        context = "\n".join(self.df.iloc[idx]['response'] for idx in indices[0])

        # Generate answer
        response = self.generate_answer(question, context)
        generated_text = response.get("response", "")

        # Compute accuracy (cosine similarity with top-1 retrieved ground truth)
        gen_embed = self.get_embedding(generated_text) if generated_text else np.zeros(768, dtype=np.float32)
        true_answer = self.df.iloc[indices[0][0]]['response']
        truth_embed = self.get_embedding(true_answer)
        similarity = cosine_similarity([gen_embed], [truth_embed])[0][0]

        return generated_text, response.get("confidence", "unknown"), similarity


if __name__ == "__main__":
    agent = CustomerSupportAgent(
        data_path="bitext/bitext_dataset.csv",
        index_path="faiss_index.bin",
        embed_cache_path="cached_embeddings.pkl"
    )

    print("\n💬 Banking Support Agent is ready! (type 'exit' to quit)\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        answer, confidence, accuracy = agent.ask_question(user_query)
        print(f"Agent: {answer}")
        print(f"Confidence: {confidence}")
        print(f"Accuracy (cosine similarity): {accuracy:.2f}\n")
