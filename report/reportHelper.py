from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

load_dotenv()


from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
open_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


import random
from typing import List, Dict
from openai import OpenAI

open_ai = OpenAI()  

def generate_questions_per_chunk(chunks: List[str], sample_size: int = 20) -> Dict[str, Dict[str, str]]:
    """
    Generate 5 types of questions for each of the sampled text chunks.
    
    Args:
        chunks (List[str]): List of all available text chunks.
        sample_size (int): Number of chunks to sample and generate questions from.
    
    Returns:
        Dict[str, Dict[str, str]]: A mapping of original chunk text to its 5 question types.
    """
    sampled_chunks = random.sample(chunks, sample_size)
    results = {}

    for chunk in sampled_chunks:
        prompt = f"""
                You are a helpful assistant. Generate 5 types of questions based on the following text.

                TEXT:
            \"\"\"{chunk}\"\"\"

            Return the questions in JSON format like:
                {{
                "short": "...",
                "detailed": "...",
                "direct": "...",
                "context_based": "...",
                "vague": "..."
                }}
                        """
        try:
            response = open_ai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            content = response.choices[0].message.content

            import json
            questions = json.loads(content)
            results[chunk] = questions
        
        except Exception as e:
            print(f"Error generating questions for chunk: {chunk[:50]}...")
            print(e)
            continue
    
    return results



from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

def simple_retrieve(
    questions: Dict[str, str],
    text_chunks: List[str],
    embedding_model,
    k: int = 3
) -> Dict[str, List[str]]:
    """
    Embed the chunks using the provided embedding_model, store in InMemoryVectorStore,
    and retrieve top-k relevant chunks for each query in the questions dictionary.
    
    Args:
        questions (Dict[str, str]): Dictionary of question types to question texts.
        text_chunks (List[str]): List of text chunks.
        embedding_model: A LangChain-compatible embedding model.
        k (int): Number of top documents to retrieve for each question.

    Returns:
        Dict[str, List[str]]: Retrieved chunks for each question type.
    """
    retrieval_results = {}

    # Convert strings to Document objects
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    # Build the vector store
    vector_store = InMemoryVectorStore.from_documents(docs, embedding_model)

    # Perform retrieval for each question
    for qtype, query in questions.items():
        results = vector_store.similarity_search(query, k=k)
        retrieval_results[qtype] = [doc.page_content for doc in results]

    return retrieval_results

import time
from typing import Dict, List, Tuple

def evaluate_overall_accuracy_and_latency(
    questions_by_chunk: Dict[str, Dict[str, str]],
    all_chunks: List[str],
    embedding_model,
    k: int = 3
) -> Tuple[float, float]:
    """
    Evaluate overall retrieval accuracy and latency.

    Args:
        questions_by_chunk: Dict mapping chunk text to its dict of question types to questions.
        all_chunks: List of all text chunks in the collection.
        embedding_model: Embedding model compatible with simple_retrieve.
        k: Number of top chunks to retrieve per question.

    Returns:
        Tuple containing:
            - overall_accuracy: float (0.0 to 1.0)
            - average_latency: float (in seconds)
    """
    total_hits = 0
    total_questions = 0
    total_latency = 0

    for chunk_text, questions_dict in questions_by_chunk.items():
        for qtype, query in questions_dict.items():
            start_time = time.time()

            retrieval_results = simple_retrieve(
                questions={qtype: query},
                text_chunks=all_chunks,
                embedding_model=embedding_model,
                k=k
            )

            end_time = time.time()
            total_latency += (end_time - start_time)
            total_questions += 1

            retrieved_chunks = retrieval_results[qtype]
            if chunk_text in retrieved_chunks:
                total_hits += 1

    overall_accuracy = total_hits / total_questions if total_questions > 0 else 0
    average_latency = total_latency / total_questions if total_questions > 0 else 0

    return overall_accuracy, average_latency
