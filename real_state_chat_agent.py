import os
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import  PreTrainedTokenizer, PreTrainedModel
from langchain.vectorstores.base import VectorStore
from embed import load_llm_model, load_db
import json

openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set
load_dotenv()


class IChatAgent(ABC):
    @abstractmethod
    def ask_real_estate_bot(self, user_query: str): any


class ChatAgent(IChatAgent):
    def __init__(self, db: VectorStore):
        self.db = db
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def ask_real_estate_bot(self, user_query : str):
        try:
            similar_docs = self.db.similarity_search(user_query, k=5)
            context = "\n\n".join([doc.page_content for doc in similar_docs])

            instructions = "You are a helpful real estate assistant for FilipinoHome."
            prompt = (
                "You are a helpful real estate assistant for FilipinoHomes.\n\n"
                "Here are some property listings:\n"
                + context +
                "\n\n"
                "User Query: " + user_query + "\n\n"
                "Based on the listings above, respond with the most relevant properties.\n\n"
                "Only provide a RFC8259 compliant JSON response following this exact format without deviation:\n"
                "[\n"
                "  {\n"
                "    \"id\": \"property id\"\n"
                "    \"title\": \"Property Title\",\n"
                "    \"url\": \"https://property-link.com\",\n"
                "    \"status\": \"For Sale or For Rent\",\n"
                "    \"category\": \"Land / Condo / House & Lot / etc.\",\n"  
                "    \"type\": \"Type of property (e.g. Commercial Lot, Studio Unit, etc.)\",\n"
                "    \"location\": \"Location of the property\",\n"
                "    \"postedOn\": \"MMM. DD, YYYY\",\n"
                "    \"price\": \"₱ amount\",\n"
                "    \"image\": \"Thumbnail image URL\"\n"
                "    \"amenities\": \"list of strings delimited by a comma\"\n"
                "  }\n"
                "]\n\n"
                "If no relevant properties are found, return an empty array: []"
            )


            print("Sending request to OpenAI...")

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.95
            )

            answer = response.choices[0].message.content
            print("Response received:\n", answer)
            # Try parsing the answer as JSON
            try:
                parsed_answer = json.loads(answer)  # Convert string to JSON if possible
                print("Parsed response:\n", parsed_answer)
                return parsed_answer
            except json.JSONDecodeError:
                print(f"Error: The response is not valid JSON: {answer}")
                raise ValueError("Received response is not a valid JSON.")


        except Exception as e:
            print(f"Error in ask_real_estate_bot: {e}")
            raise



def ChatAgentFactory() -> IChatAgent :
    db = load_db(os.getenv("VECTOR_DB"))
    return ChatAgent(db)
