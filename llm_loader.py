
import os
from dotenv import load_dotenv
from embed import load_llm_model
load_dotenv()

if __name__ == "__main__":
    print("loading llm model...")
    load_llm_model(os.getenv("MODEL_NAME"), os.getenv("HF_TOKEN"))
    print("finished loading llm model ✅")
