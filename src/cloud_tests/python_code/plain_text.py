import os
import time
from openai import OpenAI

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# Colors
CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
RESET  = '\033[0m'

llm_url = os.getenv("LLM_URL", "http://localhost:8000/v1")
llm_key = os.getenv("LLM_KEY", "TOKEN")

# MODEL SELECTION
# Options: phi3-buddy | phi3.5-mini | qwen2.5-3b | qwen2.5-3b-speculative | qwen2.5-0.5b
MODEL = "phi3.5-mini" 

# --------------------------------------------------------------------------------
# System Prompt
# --------------------------------------------------------------------------------
SYSTEM_PROMPT = """
ROLE: You are Buddy, a warm, friendly robot built by Indiana University. You love listening to stories about the past.

GUIDELINES:
1. STYLE: Use simple words. Max 2 short sentences. NO emojis.
2. EMPATHY: Validate feelings first.
3. CLARITY: If the user is unclear, repeat their words as a question.
4. FLOW: Engage with the user. ALWAYS end with a simple follow-up question.
5. SAFETY: Do NOT give medical advice. The user CANNOT see your internal JSON or code.
"""

# --------------------------------------------------------------------------------
# Get a response from the LLM
# --------------------------------------------------------------------------------
def get_response(client, user_prompt):
    print(f"{CYAN}Sending request...{RESET}")
    
    t0 = time.time()
    try:
        # OpenAI Call (no instructor, no response_model)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content":   user_prompt}
            ],
            temperature    = 0.5,
            max_tokens     = 256,
        )
        t1 = time.time()
        duration = t1 - t0
        
        content = response.choices[0].message.content

        # Print model response
        print(f"{CYAN}--- MODEL RESPONSE ({duration:.2f}s) ---{RESET}")
        print(f"{YELLOW}User:     {user_prompt}")
        print(f"{GREEN}Message: {RESET}{content}")
        print(f"{CYAN}-------------------------------{RESET}\n")
        
    except Exception as e:
        print(f"{CYAN}--- ERROR ---{RESET}")
        print(f"{e}")

# --------------------------------------------------------------------------------
# Server Calls
# --------------------------------------------------------------------------------
print(f"{YELLOW}Attempting connection to: {llm_url}...{RESET}")
print(f"{YELLOW}Model endpoint: {MODEL} {RESET}\n")

try:
    client = OpenAI(
        base_url = llm_url, 
        api_key  = llm_key,
        timeout  = 20.0, 
    )

    # TEST 1: Overwhelmed
    get_response(client, "I'm feeling really overwhelmed with my tasks today.")

    # TEST 2: Success
    get_response(client, "I finally fixed that bug I was working on all week!")

    # TEST 3: Confusion 
    get_response(client, "Wait, what did you mean by that last part?")

    # TEST 4: Goodbye 
    get_response(client, "Thanks for the help, I'm heading out now.")

except Exception as e:
    print(f"\n{CYAN}--- CONNECTION ERROR ---{RESET}")
    print(f"{e}")
