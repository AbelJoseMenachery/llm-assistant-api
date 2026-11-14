import os

import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class VectorMemory:
    def __init__(self, artifacts_dir="artifacts/current", embedding_dim=384):
        """Initializes the FAISS index and prepares for model loading."""
        self.artifacts_dir = artifacts_dir
        self.embedding_dim = embedding_dim

        # Initialize FAISS Index FlatIP
        self.index = faiss.IndexFlatIP(embedding_dim)

        # In-memory document store
        self.doc_store = {}
        self.current_id = 0

        # Pointers to active runtime models
        self.session = None
        self.tokenizer = None

        # Attempt to load models if they exist on disk at startup
        self._try_initial_load()

    def _try_initial_load(self):
        """Loads the models on startup if the artifacts exist."""
        onnx_path = os.path.join(self.artifacts_dir, "embedder_quant.onnx")

        if os.path.exists(onnx_path):
            print("Loading initial ONNX embeddings from disk...")
            self.reload_runtime(self.artifacts_dir)
        else:
            print("⚠️ No local artifacts found. Standing by for initial bootstrap sync...")

    def reload_runtime(self, new_artifacts_dir: str):
        """
        ZERO-DOWNTIME HOT RELOAD:
        Loads the new ONNX session and Tokenizer into memory BEFORE swapping.
        If loading fails, an exception is thrown and the old runtime remains perfectly active.
        """
        onnx_path = os.path.join(new_artifacts_dir, "embedder_quant.onnx")

        # Configure session options strictly for low-RAM environments (Render 512MB free tier)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = False  # Disables ONNX pre-allocating memory pools
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Load into temporary variables (Validates the artifacts natively)
        new_session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
        new_tokenizer = AutoTokenizer.from_pretrained(new_artifacts_dir)

        # Atomic Pointer Swap (Zero Downtime)
        self.session = new_session
        self.tokenizer = new_tokenizer

        print("✅ ONNX Runtime and Tokenizer successfully hot-swapped in memory!")

    def _mean_pooling(self, last_hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Averages the token embeddings to create a single sentence embedding."""
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generates L2-normalized embeddings using the active ONNX session."""
        if not self.session or not self.tokenizer:
            raise RuntimeError("Embedder model is not loaded. Artifacts are missing or currently updating.")

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")

        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64),
            'token_type_ids': inputs['token_type_ids'].astype(np.int64)
        }

        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]

        embeddings = self._mean_pooling(last_hidden_state, inputs['attention_mask'])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

        return embeddings.astype(np.float32)

    def add_texts(self, texts: list[str], metadata: list[dict] = None, batch_size: int = 2):
        """Embeds and indexes a list of text chunks incrementally using small batch size."""
        if not texts:
            return

        if metadata is None:
            metadata = [{"filename": "unknown"} for _ in texts]

        # Insert directly into FAISS inside the loop to flatline memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadata[i:i + batch_size]

            # Embed batch
            batch_embeddings = self.embed(batch_texts)

            # Index incrementally
            self.index.add(batch_embeddings)

            # Update memory store
            for text, meta in zip(batch_texts, batch_meta):
                self.doc_store[self.current_id] = {"text": text, "metadata": meta}
                self.current_id += 1

        print(f"✅ Indexed batch. Total vectors in memory: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Searches the FAISS index and returns chunks decorated with their source metadata."""
        if self.index.ntotal == 0:
            return []

        query_emb = self.embed([query])
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.doc_store:
                doc = self.doc_store[idx]
                filename = doc["metadata"].get("filename", "unknown")
                chunk_text = doc["text"]
                results.append(f"[Source: {filename}]\n{chunk_text}")
        return results
