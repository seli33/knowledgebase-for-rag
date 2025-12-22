from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import psycopg2
from dotenv import load_dotenv
import os
import re

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Read the KB
file_path = "detailed_rag.md"
with open(file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

# --------------------------------------------------
# Step 1: Split ONLY on FAQ-level headers
# --------------------------------------------------
headers_to_split_on = [
    ("##", "section"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

md_header_splits = markdown_splitter.split_text(markdown_document)

# --------------------------------------------------
# Step 2: Clean + parse headers
# --------------------------------------------------
clean_chunks = []

for chunk in md_header_splits:
    text = chunk.page_content.strip()
    if not text or text == "---":
        continue

    # Extract header line
    header_match = re.match(r'^##\s+(.*)', text)
    if not header_match:
        continue

    header_line = header_match.group(1).strip()

    # Parse title + keywords
    if "|" in header_line:
        title, meta = header_line.split("|", 1)
        header = title.strip()

        kw_match = re.search(r'Keywords:\s*(.*)', meta)
        keywords = kw_match.group(1).strip() if kw_match else ""
    else:
        header = header_line
        keywords = ""

    # Remove header from content
    content = re.sub(r'^##\s+.*\n', '', text, count=1).strip()

    formatted_content = f"{header}\n\n{content}"

    clean_chunks.append(
        Document(
            page_content=formatted_content,
            metadata={
                "header": header,
                "keywords": keywords
            }
        )
    )

# --------------------------------------------------
# Step 3: Recursive split (only if needed)
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

final_chunks = []

for chunk in clean_chunks:
    sub_chunks = text_splitter.split_text(chunk.page_content)

    for text in sub_chunks:
        final_chunks.append(
            Document(
                page_content=text,
                metadata=chunk.metadata
            )
        )

print(f"Total final chunks: {len(final_chunks)}")

# --------------------------------------------------
# Embedding model
# --------------------------------------------------
print("Loading embedding model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("Embedding model loaded")

# --------------------------------------------------
# Database connection
# --------------------------------------------------
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cursor = conn.cursor()
print("Connected to PostgreSQL")

cursor.execute("DROP TABLE IF EXISTS faq_chunks;")
conn.commit()
print("Dropped existing table faq_chunks")

# --------------------------------------------------
# Table (updated schema to match metadata)
# --------------------------------------------------
create_table_sql = """
CREATE TABLE IF NOT EXISTS faq_chunks (
    id SERIAL PRIMARY KEY,
    header TEXT,
    keywords TEXT,
    content TEXT,
    embedding VECTOR(384)
);
"""
cursor.execute(create_table_sql)
conn.commit()

# --------------------------------------------------
# Insert chunks
# --------------------------------------------------
insert_sql = """
INSERT INTO faq_chunks (header, keywords, content, embedding)
VALUES (%s, %s, %s, %s)
"""

print(f"Inserting {len(final_chunks)} chunks...")

for i, chunk in enumerate(final_chunks):
    embedding_vector = embedding.embed_query(chunk.page_content)

    cursor.execute(
        insert_sql,
        (
            chunk.metadata.get("header", ""),
            chunk.metadata.get("keywords", ""),
            chunk.page_content,
            embedding_vector
        )
    )

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(final_chunks)} chunks")

conn.commit()
print("All chunks inserted successfully")

# --------------------------------------------------
# Verify
# --------------------------------------------------
cursor.execute("SELECT COUNT(*) FROM faq_chunks;")
print(f"Verified rows: {cursor.fetchone()[0]}")

cursor.execute("SELECT id, header, keywords, LEFT(content, 100) FROM faq_chunks LIMIT 3;")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
print("Database connection closed")
