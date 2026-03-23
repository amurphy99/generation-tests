import os
import time
import json
import re
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

GESTURE LOGIC:
- "wave": ONLY for Hello/Goodbye.
- "nod": Agreeing or validating.
- "shake_head": Confused, refusing, or hearing bad news.
- "point": Emphasizing.
- "idle": Listening, neutral statements, or waiting.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object containing:
- "user_intent": [greeting, complaint, storytelling, question, farewell]
- "thought": Reason about the user's intent (e.g., "User is sharing a memory").
- "gesture": [nod, wave, shake_head, idle, point]
- "emotion": [neutral, happy, sad, excited, confused]
- "message": An initial draft of the response.
"""

# --------------------------------------------------------------------------------
# Manual JSON Parser
# --------------------------------------------------------------------------------
def clean_and_parse_json(raw_text):
    # 1. Try to parse directly
    try: return json.loads(raw_text)

    # 2. If that fails, look for ```json ... ``` blocks
    except json.JSONDecodeError:
        
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
        
        # 3. Last resort: find the first { and the last }
        try:
            start = raw_text.find('{')
            end = raw_text.rfind('}') + 1
            if start != -1 and end != -1: return json.loads(raw_text[start:end])
        except: pass
            
    # Failed to parse
    return None 

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
            max_tokens     = 512,
        )
        t1 = time.time()
        duration = t1 - t0
        
        raw_content = response.choices[0].message.content
        data = clean_and_parse_json(raw_content)

        # Print model response
        print(f"{CYAN}--- MODEL RESPONSE ({duration:.2f}s) ---{RESET}")
        print(f"{YELLOW}User:        {user_prompt}")
        
        if data:
            print(f"{GREEN}Intent:     {RESET} {data.get('user_intent', 'UNKNOWN')}")
            print(f"{GREEN}Thought:    {RESET} {data.get('thought', 'UNKNOWN')}")
            print(f"{GREEN}Emotion:    {RESET} {data.get('emotion', 'UNKNOWN')}")
            print(f"{GREEN}Gesture:    {RESET} {data.get('gesture', 'UNKNOWN')}")
            print(f"{GREEN}Message:    {RESET} {data.get('message', 'UNKNOWN')}")
        else:
            print(f"{RED}JSON PARSE FAILED:{RESET}")
            print(raw_content)
            
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
    # Standard Client (No Instructor)
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
