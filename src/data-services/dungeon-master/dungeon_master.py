import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re

load_dotenv()

# Initialize the client
client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

def get_system_prompt(filename="src/docs/config.json"):
    """Reads the system prompt from a JSON file."""
    with open(filename, "r") as f:
        config = json.load(f)
    return config.get("system_prompt", "You are a Dungeon Master.")

def call_dungeon_master(player_input):
    system_message = get_system_prompt()

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1:novita",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": player_input}
        ]
    )

    raw_content = completion.choices[0].message.content

    # Remove DeepSeek thinking block
    if "</think>" in raw_content:
        raw_content = raw_content.split("</think>")[-1].strip()

    try:
        # Extract JSON from potential markdown code blocks
        json_str = re.search(r"\{.*\}", raw_content, re.DOTALL).group(0)
        data = json.loads(json_str)
        return data.get("DM", "The  mists swallow your words... (Error parsing DM response)")
    except (json.JSONDecodeError, AttributeError):
        # Fallback if the model fails to provide valid JSON
        return raw_content
    
# --- Example Usage ---
if __name__ == "__main__":
    response = call_dungeon_master("I enter the dark tavern and look for a hooded figure in the corner.")
    print(f"DM: {response}")