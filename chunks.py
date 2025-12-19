from langchain_text_splitters import MarkdownHeaderTextSplitter ,RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
import psycopg2
from dotenv import load_dotenv
import os

# Basic connection string
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

#Read the kb
file_path ="detailed_rag.md"
with open(file_path,"r",encoding="utf-8")as f:
    markdown_document=f.read()


# Step 1: Split by markdown headers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Step 2: Clean chunks
clean_chunks = [
    c for c in md_header_splits
    if c.page_content.strip() not in ["", "---"]
]
# Step 2: Clean chunks
for chunk in clean_chunks:
    header = chunk.metadata.get("Header 2", "")
    chunk.page_content = f"{header}\n\n{chunk.page_content}"

# Step 4: Further split large chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

final_chunks = []
for chunk in clean_chunks:
    # Use split_text() for strings, not split_documents()
    sub_chunks = text_splitter.split_text(chunk.page_content)
    for text in sub_chunks:
        final_chunks.append(
            Document(page_content=text, metadata=chunk.metadata)
        )

# Display results
print(f"Total final chunks: {len(final_chunks)}")
for i, chunk in enumerate(final_chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:200]}...")

#Embedding model
print("Loading HuggingFace model (first time may take a moment)...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)

print("Embedding model loaded")
#databse connect
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = conn.cursor()
print("✅ Connected to PostgreSQL")


# Step 7: Create table if it doesn't exist (with proper column order matching your schema)
create_table_sql = """
CREATE TABLE IF NOT EXISTS faq_chunks (
    id SERIAL PRIMARY KEY,
    header TEXT,
    subheader TEXT,
    content TEXT,
    embedding VECTOR(384)
);
"""
cursor.execute(create_table_sql)
conn.commit()
print("Table 'faq_chunks' ready")


print(f"\nInserting {len(final_chunks)} chunks into database...")

insert_sql="""
INSERT INTO faq_chunks (header, subheader, content, embedding)
VALUES (%s, %s, %s, %s)
"""

# Insert each chunk with its embedding
for i, chunk in enumerate(final_chunks):
    # Generate embedding for this chunk
    embedding_vector = embedding.embed_query(chunk.page_content)
    
    # Extract metadata
    header = chunk.metadata.get('Header 1', '')
    subheader = chunk.metadata.get('Header 2', '')
    content = chunk.page_content
    
     # Insert into database
    cursor.execute(insert_sql, (
        header,
        subheader,
        content,
        embedding_vector  # List of 384 floats
    ))
    
    if i % 10 == 0:  # Progress indicator
        print(f"  Processed {i+1}/{len(final_chunks)} chunks")

conn.commit()
print(f"Inserted {len(final_chunks)} chunks into database")

#  Verify insertion
cursor.execute("SELECT COUNT(*) FROM faq_chunks;")
count = cursor.fetchone()[0]
print(f"Verified: {count} rows in database")

# Display sample
cursor.execute("SELECT id, header, subheader, LEFT(content, 100) FROM faq_chunks LIMIT 3;")
print("\n Sample entries:")
for row in cursor.fetchall():
    print(f"  ID: {row[0]}")
    print(f"  Header: {row[1]}")
    print(f"  Subheader: {row[2]}")
    print(f"  Content: {row[3]}...")
    print()

# Close connection
cursor.close()
conn.close()
print("Database connection closed")