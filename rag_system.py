from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
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
            model_name="sentence-transformers/all-MiniLM-L6-v2",
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
        print("Initializing llm...")
        self.llm = Ollama(
        model="gemma2",             
        base_url="http://localhost:11434",  # optional, default
        temperature=0.2
    )
        print("RAG System ready!\n")
    
    def retrieve_similar_chunks(self, query, top_k=5):
        # Generate embedding for the query
        query_embedding = self.embedding.embed_query(query)
        
        # Perform similarity search using cosine distance
        search_sql = """
        SELECT 
            id,
            header,
            subheader,
            content,
            embedding <=> %s::vector AS distance
        FROM faq_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """
        
        self.cursor.execute(search_sql, (query_embedding, query_embedding, top_k))
        results = self.cursor.fetchall()
        
        return results
    
    def format_context(self, chunks):
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            id_, header, subheader, content, distance = chunk
            context_parts.append(
                f"[Document {i}]\n"
                f"Header: {header}\n"
                f"Subheader: {subheader}\n"
                f"Content: {content}\n"
                f"Relevance Score: {1 - distance:.3f}\n"
            )
        return "\n".join(context_parts)
    
    def generate_answer(self, query, context):
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- If the context doesn't contain enough information to answer the question, say I dont know
- Be concise and clear in your response
- Cite which document(s) you're using if relevant

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response
    
    def query(self, question, top_k=3, show_context=False):
        print("Question:", question, "\n")
        
        # Step 1: Retrieve similar chunks
        print("Retrieving top", top_k, "relevant chunks...")
        chunks = self.retrieve_similar_chunks(question, top_k)
        
        if not chunks:
            print("No relevant information found in the database")
            return None
        
        print("Found", len(chunks), "relevant chunks\n")
        
        # Show retrieved chunks if requested
        if show_context:
            print("Retrieved Context:")
            print("-" * 80)
            for i, chunk in enumerate(chunks, 1):
                id_, header, subheader, content, distance = chunk
                print("\n[Chunk", i, "] (Similarity:", f"{1-distance:.3f})")
                print("Header:", header)
                print("Subheader:", subheader)
                print("Content:", content[:200], "...")
            print("\n" + "-" * 80 + "\n")
        
        # Step 2: Format context
        context = self.format_context(chunks)
        
        # Step 3: Generate answer
        print("Generating answer...")
        answer = self.generate_answer(question, context)
        
        print("Answer:")
        print(answer)
        print("\n" + "=" * 80 + "\n")
        
        return answer
    
    def close(self):
        self.cursor.close()
        self.conn.close()
        print("Database connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem()
    
    # Interactive mode
    print("RAG System Interactive Mode")
    print("Type 'quit' to exit, 'context' to toggle context display\n")
    
    show_context = False
    
    while True:
        user_input = input("Ask a question: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if user_input.lower() == 'context':
            show_context = not show_context
            print("Context display:", "ON" if show_context else "OFF", "\n")
            continue
        
        if not user_input:
            continue
        
        rag.query(user_input, top_k=5, show_context=show_context)
    
    # Close connection
    rag.close()