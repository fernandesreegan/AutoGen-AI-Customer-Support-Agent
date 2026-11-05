import os
import faiss
from sentence_transformers import SentenceTransformer

class KnowledgeBaseRAG:
    def __init__(self, kb_dir="knowledge_base", index_file="faiss_index.bin"):
        self.kb_dir = kb_dir
        self.index_file = index_file
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.metadata = []
        self.index = None
        self.load_and_index()

    def load_and_index(self):
        # Read text from markdown files
        for filename in os.listdir(self.kb_dir):
            if filename.endswith(".md"):
                path = os.path.join(self.kb_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    sections = content.split("\n\n")
                    for section in sections:
                        if section.strip():
                            self.documents.append(section.strip())
                            self.metadata.append({"source": filename})

        if not self.documents:
            return

        # Check if FAISS index already exists on disk
        if os.path.exists(self.index_file):
            # Load existing FAISS index from disk
            self.index = faiss.read_index(self.index_file)
        else:
            # Create new embeddings and FAISS index
            embeddings = self.model.encode(self.documents)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Save the index to disk for future runs
            faiss.write_index(self.index, self.index_file)

    def search(self, query, k=2):
        if not self.index:
            return []
        
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append({
                    "passage": self.documents[idx],
                    "source": self.metadata[idx]["source"]
                })
        return results