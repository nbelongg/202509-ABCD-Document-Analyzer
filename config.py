import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

possible_models = json.loads(os.getenv("possible_models"))

API_KEY = os.getenv("API_KEY")

API_SECRET = os.getenv("API_SECRET")