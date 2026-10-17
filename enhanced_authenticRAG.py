import os
import pythainlp
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

class EnhancedAuthenticRAG:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")
        
        print("กำลังโหลดโมเดล AI กรุณารอสักครู่...")
        # ใช้ BGE-M3
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.opensearch_client = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}], use_ssl=False)
        self.vector_index = "enhanced-vector-index"
        self.bm25_index = "enhanced-bm25-index"
        
        # ตัดคำภาษาไทยแบบประโยคไม่ขาด
        self.sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100, tokenizer=pythainlp.tokenize.word_tokenize)
        self._create_indices()

    def _create_indices(self):
        # สร้าง Vector Index
        if not self.opensearch_client.indices.exists(index=self.vector_index):
            self.opensearch_client.indices.create(index=self.vector_index, body={
                "settings": {"index.knn": True},
                "mappings": {
                    "properties": {
                        "embedding": {"type": "knn_vector", "dimension": 1024},
                        "content": {"type": "text", "analyzer": "thai"},
                        "contextualized_content": {"type": "text", "analyzer": "thai"}
                    }
                }
            })
        
        # สร้าง BM25 Index (Thai Analyzer)
        if not self.opensearch_client.indices.exists(index=self.bm25_index):
            self.opensearch_client.indices.create(index=self.bm25_index, body={
                "settings": {"analysis": {"analyzer": {"default": {"type": "thai"}}}},
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "thai"},
                        "contextualized_content": {"type": "text", "analyzer": "thai"}
                    }
                }
            })

    def generate_context(self, full_doc, chunk):
        prompt = f"<document>{full_doc}</document>\n<chunk>{chunk}</chunk>\nPlease give a short succinct context in Thai."
        try:
            res = self.client.chat.completions.create(model="qwen/qwen-2.5-72b-instruct", messages=[{"role": "user", "content": prompt}], max_tokens=100)
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating context: {e}")
            return "ไม่มีบริบทเพิ่มเติม"

    def ingest_documents(self, md_paths):
        llama_docs = []
        full_text_list = []
        
        # ใช้คำสั่งอ่านไฟล์แบบเบสิกของ Python แทน Langchain
        for path in md_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    full_text_list.append(content)
                    llama_docs.append(Document(text=content, metadata={"source": path}))
        
        if not llama_docs:
            print("ไม่พบไฟล์เอกสารใน corpus_input เลยครับ! กรุณาเช็คว่าใส่ไฟล์ 1.md, 2.md ไว้ในโฟลเดอร์แล้ว")
            return

        full_text = "\n".join(full_text_list)
        chunks = self.sentence_splitter.get_nodes_from_documents(llama_docs)
        
        print(f"กำลังเริ่มสร้าง Context และนำเข้าข้อมูลทั้งหมด {len(chunks)} Chunks...")
        for i, chunk in enumerate(chunks):
            ctx = self.generate_context(full_text, chunk.text)
            ctx_content = f"บริบท: {ctx}\nเนื้อหา: {chunk.text}"
            
            # ทำ L2 Normalization ก่อนนำเข้า Vector
            vector = self.embed_model.encode(ctx_content, normalize_embeddings=True).tolist()
            
            self.opensearch_client.index(index=self.vector_index, id=str(i), body={
                "embedding": vector, "content": chunk.text, "contextualized_content": ctx_content
            })
            self.opensearch_client.index(index=self.bm25_index, id=str(i), body={
                "content": chunk.text, "contextualized_content": ctx_content
            })
            print(f"บันทึก Chunk ที่ {i+1}/{len(chunks)} เรียบร้อย")
            
        print("นำเข้าข้อมูลเสร็จสมบูรณ์ 100%!")

if __name__ == "__main__":
    rag = EnhancedAuthenticRAG()
    # ดึงไฟล์ที่อยู่ในโฟลเดอร์ corpus_input
    rag.ingest_documents(["./corpus_input/1.md", "./corpus_input/2.md", "./corpus_input/44.md", "./corpus_input/5555.md"])