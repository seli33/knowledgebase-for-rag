from langchain_text_splitters import MarkdownHeaderTextSplitter ,RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

# Step 3: Add header to content
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