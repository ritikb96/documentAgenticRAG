import re
import os
import asyncio
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI



try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

List[float]


class SimpleSemanticChunker:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.6,
        min_chunk_sentences=3,
        max_chunk_sentences=10,
        overlap=1,
        chunking_method = 'semantic_chuking'

    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.overlap = overlap
        self.chunking_method = chunking_method

    def preprocess(self, text: str) -> str:
        # Normalize whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def split_paragraphs(self, text: str) -> List[str]:
        # Split on double newlines or known section headers 
        paragraphs = re.split(r'\n{2,}', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    def compute_similarity(self, sentences: List[str]) -> List[float]:
        embeddings = self.model.encode(sentences)
        adjacent_similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]
        return adjacent_similarities

    def find_boundaries(self, adjacent_similarities: List[float], sentences: List[str]) -> List[int]:
        boundaries = [0]
        sentence_count = 0

        for i, sim in enumerate(adjacent_similarities, start=1):
            sentence_count += 1
            # Split if similarity below threshold AND min size reached, OR max chunk size reached
            if (sim < self.similarity_threshold and sentence_count >= self.min_chunk_sentences) or (sentence_count >= self.max_chunk_sentences):
                boundaries.append(i)
                sentence_count = 0

        boundaries.append(len(sentences))
        return boundaries

    def chunk(self, text: str) -> List[str]:
        text = self.preprocess(text)
        paragraphs = self.split_paragraphs(text)

        all_sentences = []
        para_boundaries = []  
        count = 0
        for para in paragraphs:
            sents = self.split_sentences(para)
            all_sentences.extend(sents)
            count += len(sents)
            para_boundaries.append(count)

        if not all_sentences:
            return []

        # Compute similarities between sentences
        sim_matrix = self.compute_similarity(all_sentences)

        # Find chunk boundaries respecting similarity threshold and chunk size limits
        boundaries = self.find_boundaries(sim_matrix, all_sentences)

        # Adjust boundaries: only snap to paragraph boundary if within 2 sentences of a boundary, to avoid big merges
        adjusted_boundaries = [boundaries[0]]
        for b in boundaries[1:-1]:
            close_paras = [pb for pb in para_boundaries if abs(pb - b) <= 2 and pb > adjusted_boundaries[-1]]
            if close_paras:
                adjusted_boundaries.append(min(close_paras))
            else:
                adjusted_boundaries.append(b)
        adjusted_boundaries.append(boundaries[-1])

        chunks = []
        for i in range(len(adjusted_boundaries) - 1):
            start = adjusted_boundaries[i]
            end = adjusted_boundaries[i + 1]
            if i > 0 and self.overlap > 0:
                start = max(start - self.overlap, 0)
            chunk_text = ' '.join(all_sentences[start:end])
            chunks.append(chunk_text)
        print("Chunking done")

        return {
            "chunks":chunks,
            "chunking_method":self.chunking_method
        }


# helper function
def semantic_chunk_text(text: str, threshold=0.6, min_chunk_sentences=3, max_chunk_sentences=10, overlap=1) -> Dict[str,Any]:
    try:
        chunker = SimpleSemanticChunker(
        similarity_threshold=threshold,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
        overlap=overlap
        )
        return chunker.chunk(text)
    except Exception as e:
        print(f"Chunking failed: {e}")
        return None
    






