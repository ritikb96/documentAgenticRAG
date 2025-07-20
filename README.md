# 📄 Document Agentic RAG

An advanced agent-based Retrieval-Augmented Generation (RAG) system designed to process and retrieve knowledge from uploaded documents with high accuracy and flexibility. Built using modern LLM tooling and an orchestration-first approach, this system combines the power of OpenAI GPT-4 Omni with custom chunking, semantic embeddings, and multi-agent routing.

---

## ⚙️ Tech Stack Overview

- **API Framework**: FastAPI
- **Document Parsing**: [`docling`](https://github.com/langchain-ai/docling)
- **LLM**: OpenAI `gpt-4-omni`
- **Embedding Model**: `text-embedding-3-small`
- **Chunking Strategy**: Custom chunking based on semantic overlap and cosine similarity
- **Vector Store**: Qdrant
- **Relational DB**: Supabase (PostgreSQL)
- **Agent Framework**: LangGraph + LangChain
- **Tool Management**: LangSmith
- **Agent Memory**: Redis
- **Email Interview Booking**: Agent-triggered SMTP-based booking

---

## 🤖 Agentic Architecture

The system uses LangGraph to coordinate multiple agent nodes:
- **Router**: Determines if the query is RAG-based, search-based, or booking-related.
- **RAG Lookup**: Retrieves vector matches from Qdrant.
- **Web Search**: Executes a web-based search when vector results are insufficient.
- **Answer Generator**: LLM response generation node.
- **Booking Tool**: Handles interview booking and sends emails via SMTP.

Each node is monitored via LangSmith for tool insights and debugging.

---

## 📊 Evaluation Report

The performance of the system’s embedding and chunking strategies has been documented in detail, including metrics such as:
- **Retrieval Accuracy**
- **Latency**
- **Chunking Efficiency**

Find the full evaluation in:  
📁 `report/report.ipynb`

---

## 📬 Booking System

Users can book interviews through the conversational agent. The agent gathers details (name, date, time, email) and stores the booking in Supabase, while confirming via email using SMTP.

---

## 🚀 Getting Started

Coming soon: setup instructions, API endpoints, and deployment notes.

---

## 📎 License

MIT – feel free to use, modify, and contribute!
