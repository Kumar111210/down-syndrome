# test_env.py
import os
print("VERY EARLY DEBUG - OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("Current working dir at start:", os.getcwd())
from dotenv import load_dotenv
import os

load_dotenv()
print("OPENAI_API_KEY =", repr(os.getenv("OPENAI_API_KEY")))