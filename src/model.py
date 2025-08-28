# import os
# import time
# import json
# import pickle
# import numpy as np
# import pandas as pd
# import faiss
# from dotenv import load_dotenv
# from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
# from google.generativeai import GenerativeModel, configure, embed_content, get_model
# from google.api_core import exceptions as google_exceptions
# from google.generativeai import embed_content

# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, wait_random_exponential

# class CustomerSupportAgent:
#     def __init__(self, data_path="bitext/bitext_dataset.csv", sample_size=200, index_path="faiss_index.bin", embeddings_cache_path="cached_embeddings.pkl"):
#         load_dotenv()
#         self.api_key = os.getenv("GOOGLE_API_KEY")
#         if not self.api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables.")

#         configure(api_key=self.api_key)
#         # Use the same embedding model as in get_embedding
#         self.embedding_model_name = 'models/text-embedding-004'
#         self.model = GenerativeModel('gemini-1.5-flash')

#         self.df = None # Initialize df to None, will be loaded if index is not found
#         self.cached_embeddings = {}
#         self.index = None
#         self.responses = []
#         self.index_path = index_path
#         self.embeddings_cache_path = embeddings_cache_path

#         # Try to load existing index and cache
#         if os.path.exists(self.index_path) and os.path.exists(self.embeddings_cache_path):
#             try:
#                 print("Loading existing FAISS index and embeddings cache...")
#                 self.index = faiss.read_index(self.index_path)
#                 with open(self.embeddings_cache_path, "rb") as f:
#                     self.cached_embeddings = pickle.load(f)
#                 # Assuming responses are stored separately or can be reconstructed
#                 # For this model, responses are derived from the loaded df
#                 # We need to load the df to get the responses
#                 data_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
#                 self.df = self._load_dataset(data_full_path, sample_size=None) # Load full dataset to get all responses
#                 self.responses = self.df['response'].tolist()
#                 print("Successfully loaded existing knowledge base.")
#             except Exception as e:
#                 print(f"Error loading existing knowledge base: {e}")
#                 self._build_new_knowledge_base(data_path, sample_size)
#         else:
#             print("Existing knowledge base not found. Building new one...")
#             self._build_new_knowledge_base(data_path, sample_size)

#         self.few_shot_examples = [
#             {"q": "How do I reset my password?", "a": "To reset your password, go to the login page and click 'Forgot Password'."},
#             {"q": "Where can I find my billing information?", "a": "Billing information is available under 'My Account' > 'Billing'."},
#             {"q": "Can I change my email address?", "a": "Yes, navigate to settings and update your email under 'Account Info'."},
#         ]
#         self.example_text = "\n".join([f"User: {ex['q']}\nAgent: {ex['a']}" for ex in self.few_shot_examples])
#         self.output_format_instruction = "Please answer in JSON format with fields 'response' and 'confidence'. Example: {\"response\": \"Your answer here.\", \"confidence\": \"high\"}"

#     def _build_new_knowledge_base(self, data_path, sample_size):
#         data_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)
#         self.df = self._load_dataset(data_full_path, sample_size)
#         self.cached_embeddings = {}
#         self.index = self._build_faiss_index()
#         self.responses = self.df['response'].tolist()
#         self.save_index_and_cache() # Save the newly built index and cache

#     def _load_dataset(self, path, sample_size):
#         df = pd.read_csv(path)
#         if {'instruction', 'response'}.issubset(df.columns):
#             df = df.rename(columns={"instruction": "prompt"})
#         df = df.dropna(subset=["prompt", "response"])
#         if "intent" in df.columns:
#             df["intent"] = df["intent"].fillna("unknown")
#         if sample_size is not None and sample_size < len(df):
#              return df.sample(n=sample_size, random_state=42)
#         return df # Return full dataframe if sample_size is None or larger than dataset

#     @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
#     def get_embedding(self, text):
#         # Ensure embedding model is configured
#         # configure(api_key=self.api_key) # Should be configured in __init__
        
#         if text not in self.cached_embeddings:
#             response = embed_content(
#                 model=self.embedding_model_name,
#                 content=text,
#                 task_type='retrieval_document'
#             )
#             self.cached_embeddings[text] = np.array(response['embedding'], dtype=np.float32)
#         return self.cached_embeddings[text]

#     def _build_faiss_index(self):
#         embeddings = np.array([self.get_embedding(text) for text in tqdm(self.df['prompt'])], dtype=np.float32)
#         index = faiss.IndexFlatL2(embeddings.shape[1])
#         index.add(embeddings)
#         return index

#     @retry(
#         stop=stop_after_attempt(3),
#         wait=wait_exponential(multiplier=2, min=10, max=300),
#         retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
#     )
#     def generate_answer(self, question, context):
#         prompt = f"""**Customer Support Response Guidelines**
# Context Information:
# {context}

# Example Interactions:
# {self.example_text}

# Current Query:
# {question}

# Response Requirements:
# 1. Provide clear, step-by-step instructions
# 2. Use markdown formatting for lists
# 3. Maximum 3 sentences
# 4. If uncertain, offer to escalate

# Required JSON Format:
# {self.output_format_instruction}"""

#         try:
#             response = self.model.generate_content(prompt)
#             if not response.text:
#                 return {"response": "Please wait while I connect you to a specialist.", "confidence": "low", "source": "fallback"}
#             try:
#                 return json.loads(response.text.strip("```json\n").strip("```"))
#             except json.JSONDecodeError:
#                 return {"response": response.text, "confidence": "medium", "source": "direct"}
#         except google_exceptions.ResourceExhausted as e:
#             print("Quota exceeded, retrying in 5 mins.")
#             time.sleep(300)
#             raise
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return {"response": "Apologies, we're experiencing technical difficulties.", "confidence": "low", "source": "error"}

#     def process_single_query(self, row):
#         query = row['prompt']
#         true_answer = row['response']
#         query_vec = self.get_embedding(query).reshape(1, -1)
#         _, indices = self.index.search(query_vec, k=3)
#         context = "\n".join(self.df.iloc[idx]['response'] for idx in indices[0])
#         gen_output = self.generate_answer(query, context)
#         gen_text = gen_output.get('response', '')
#         gen_embed = self.get_embedding(gen_text) if gen_text else np.zeros(768, dtype=np.float32)
#         truth_embed = self.get_embedding(true_answer)
#         similarity = cosine_similarity([gen_embed], [truth_embed])[0][0]
#         return {
#             "query": query,
#             "generated": gen_output,
#             "ground_truth": true_answer,
#             "similarity": similarity
#         }

#     def process_dataset(self, delay=10):
#         results = []
#         for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Queries"):
#             try:
#                 result = self.process_single_query(row)
#             except Exception as e:
#                 result = {
#                     "query": row.get("prompt", ""),
#                     "generated": {"response": "Error", "confidence": "low"},
#                     "ground_truth": row.get("response", ""),
#                     "similarity": 0.0
#                 }
#             results.append(result)
#             time.sleep(delay)
#         return results

#     def analyze_results(self, results):
#         sorted_results = sorted(results, key=lambda x: x['similarity'], reverse=True)
#         counts = {'low': 0, 'medium': 0, 'high': 0}

#         for res in sorted_results:
#             sim = res['similarity']
#             if sim < 0.5:
#                 counts['low'] += 1
#             elif sim < 0.7:
#                 counts['medium'] += 1
#             else:
#                 counts['high'] += 1

#         print("Top 5 Responses:")
#         for res in sorted_results[:5]:
#             print(f"Query: {res['query']}\nGenerated: {res['generated']['response']}\nSimilarity: {res['similarity']:.2f}\n")

#         total = len(sorted_results)
#         print("Final Stats:")
#         for k in counts:
#             print(f"{k.capitalize()} similarity: {counts[k]} ({(counts[k]/total)*100:.2f}%)")
#         return sorted_results

#     def simulate_chat(self, user_inputs):
#         for user_msg in user_inputs:
#             print(f"\n\033[1mUser:\033[0m {user_msg}")
#             try:
#                 self.test_custom_query(user_msg)
#             except Exception:
#                 print("\033[1mAgent:\033[0m Sorry, an error occurred.")

    

#     # def test_custom_query(self, question):
#     #     query_vec = self.get_embedding(question).reshape(1, -1)
#     #     _, indices = self.index.search(query_vec, k=3)
#     #     context = "\n".join(self.df.iloc[idx]['response'] for idx in indices[0])
#     #     response = self.generate_answer(question, context)
#     #     print("\033[1mAgent:\033[0m", response.get("response", ""))
#     #     print("Confidence:", response.get("confidence", "unknown"))

#     # def save_index_and_cache(self):
#     #     faiss.write_index(self.index, self.index_path)
#     #     with open(self.embeddings_cache_path, "wb") as f:
#     #         pickle.dump(self.cached_embeddings, f)
#     #     print("Saved FAISS index and cached embeddings.")

import os
import pandas as pd
import faiss
import pickle
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini
genai.configure(api_key="AIzaSyCOemZ83Y3OILpTH6I7dFoy9cd4EpJryxs")

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
