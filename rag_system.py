from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import psycopg2
from dotenv import load_dotenv
import os

# Configuration
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


class RAGSystem:
    def __init__(self):
        print("Initializing RAG System...")

        # Load embedding model
        print("Loading embedding model...")
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded")

        # Connect to database
        print("Connecting to database...")
        self.conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        self.cursor = self.conn.cursor()
        print("Database connected")

        # Initialize LLM
        self.llm = Ollama(
            model="qwen2.5:1.5b",
            temperature=0.1,
            num_ctx=2048,
            num_predict=200,
            num_thread=2,
        )

        print("RAG System ready!\n")

    # ---------------- RETRIEVAL ----------------
    def retrieve_similar_chunks(self, query, top_k=5):
        query_embedding = self.embedding.embed_query(query)

        search_sql = """
        SELECT 
            id,
            header,
            keywords,
            content,
            embedding <=> %s::vector AS distance
        FROM faq_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        self.cursor.execute(search_sql, (query_embedding, query_embedding, top_k))
        results = self.cursor.fetchall()

        # -------- Intent-aware filtering (keyword-based) --------
        query_lower = query.lower()

        if any(word in query_lower for word in ["eligib", "eligible", "apply", "requirement", "criteria"]):
            filtered = [
                r for r in results
                if r[2] and "eligib" in r[2].lower()
            ]
            if filtered:
                results = filtered

        return results

    # ---------------- CONTEXT FORMATTING ----------------
    def format_context(self, chunks):
        context_parts = []
        MAX_CHARS = 600

        for i, chunk in enumerate(chunks, 1):
            id_, header, keywords, content, distance = chunk
            context_parts.append(
                f"[Document {i}]\n"
                f"Question: {header}\n"
                f"Answer:\n{content[:MAX_CHARS]}\n"
            )

        return "\n".join(context_parts)

    # ---------------- GENERATION ----------------
    def generate_answer(self, query, context):
        prompt = f"""
You are a retrieval-based FAQ assistant for the AI Fellowship Program.

Your job is to answer the user's question using ONLY the information explicitly stated
in the provided context.

STRICT RULES:
- Use ONLY the context below.
- Use formal and friendly tone for the students.
- Do NOT use prior knowledge or assumptions.
- You can add explanations, examples, or extra details.
- If a section labeled "Quick Answer" exists in the context, you MUST use it as the primary source of the answer.Ignore lists of questions, variations, or related topics.
- You MAY combine information from multiple documents if they answer different parts of the same question.
- If the context does NOT clearly and directly answer the question, respond EXACTLY with:
"This information is not available in the FAQ."

Answering instructions:
- If the question has multiple parts, answer only the parts supported by the context.
- If ANY part cannot be answered, return the fallback sentence.
- Prefer direct sentences taken from the context.
- Keep the answer concise and factual.

Context:
{context}

Question:
{query}

Answer:
"""
        return self.llm.invoke(prompt)

    # ---------------- MAIN QUERY PIPELINE ----------------
    def query(self, question, top_k=5, show_context=False):
        print("Question:", question, "\n")

        print("Retrieving top", top_k, "relevant chunks...")
        chunks = self.retrieve_similar_chunks(question, top_k)

        if not chunks:
            return "This information is not available in the FAQ."

        print("Found", len(chunks), "relevant chunks\n")

        # Debug view
        if show_context:
            print("Retrieved Context:")
            print("-" * 80)
            for i, chunk in enumerate(chunks, 1):
                id_, header, keywords, content, distance = chunk
                print(f"\n[Chunk {i}] (Similarity: {1 - distance:.3f})")
                print("Header:", header)
                print("Keywords:", keywords)
                print("Content:", content[:200], "...")
            print("\n" + "-" * 80 + "\n")

        # -------- Similarity Gate (ANTI-HALLUCINATION) --------
        similarities = [1 - chunk[4] for chunk in chunks]
        max_similarity = max(similarities)

        MIN_SIMILARITY = 0.5
        if max_similarity < MIN_SIMILARITY:
            return "This information is not available in the FAQ."

        # Format context
        context = self.format_context(chunks)

        # Generate answer
        answer = self.generate_answer(question, context)

        # -------- Post-answer normalization --------
        if "not available in the faq" in answer.lower():
            answer = "This information is not available in the FAQ."

        return answer

    # ---------------- CLEANUP ----------------
    def close(self):
        self.cursor.close()
        self.conn.close()
        print("Database connection closed")


# ---------------- RUN MODE ----------------
if __name__ == "__main__":
    rag = RAGSystem()

    print("RAG System Interactive Mode")
    print("Type 'quit' to exit, 'context' to toggle context display\n")

    show_context = True

    while True:
        user_input = input("Ask a question: ").strip()

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'context':
            show_context = not show_context
            print("Context display:", "ON" if show_context else "OFF", "\n")
            continue

        if not user_input:
            continue

        print("\nAnswer:")
        print(rag.query(user_input, top_k=5, show_context=show_context))
        print("\n" + "=" * 80 + "\n")

    rag.close()
