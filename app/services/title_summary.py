import os
import json
import asyncio
from typing import Dict
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChunkSummarizer:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.client = openai_client

    async def get_title_and_summary(self, chunk: str) -> Dict[str, str]:
        system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
                        Return a JSON object with 'title' and 'summary' keys.
                        For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
                        For the summary: Create a concise summary of the main points in this chunk.
                        Keep both title and summary concise but informative."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document chunk:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {
                "title": "Error processing title",
                "summary": "Error processing summary"
            }
