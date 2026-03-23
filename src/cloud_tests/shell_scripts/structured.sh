#!/bin/bash

# ================================================================================
# Configuration
# ================================================================================
# LLM server IP & nginx authorization key
#TARGET_URL=
#TARGET_KEY=

# Styling Variables
BOLD='\033[1m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper function for headers
function log_step() {
    echo -e "${BLUE}------------------------------------------------------------${NC}"
    echo -e "${BOLD}${CYAN}STEP $1:${NC} ${GREEN}$2${NC}"
    echo -e "${BLUE}------------------------------------------------------------${NC}"
}

# ================================================================================
# 1. Connection Check
# ================================================================================
log_step "1" "Testing Connection to Server ($TARGET_URL)"
echo -e "${YELLOW}Pinging server models endpoint...${NC}"

if curl -H "Authorization: Bearer $TARGET_KEY" --connect-timeout 3 -s "$TARGET_URL/models" > /dev/null; then
    echo -e "${GREEN}SUCCESS: Server is reachable!${NC}"
else
    echo -e "${RED}ERROR: Cannot reach $TARGET_URL${NC}"
    echo -e "${RED}Possible causes:${NC}"
    echo -e "1. The Nginx key is incorrect."
    echo -e "2. The server on 10.128.0.20 is not running."
    echo -e "3. A firewall is blocking port 8080."
    echo -e "4. The server is still busy generating a huge response (Restart the GPU container!)."
    echo -e "5. The server IP/Port is incorrect."
    exit 1
fi

# ================================================================================
# 2. Setup Environment
# ================================================================================
log_step "2" "Setting up build environment"
mkdir -p struct_bot_build
cd struct_bot_build

# ================================================================================
# 3. Create Python Script
# ================================================================================
log_step "3" "Generating Python client code (main.py)"
cat << 'EOF' > main.py
import os
import time
import instructor

from pydantic import BaseModel, Field
from typing   import Literal
from openai   import OpenAI

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# Colors
CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RESET  = '\033[0m'

llm_url = os.getenv("LLM_URL", "http://localhost:8000/v1")
llm_key = os.getenv("LLM_KEY", "TOKEN")

# MODEL SELECTION
# Options: phi3-buddy | phi3.5-mini | qwen2.5-3b | qwen2.5-3b-speculative | qwen2.5-0.5b
MODEL = "phi3.5-mini" 

# --------------------------------------------------------------------------------
# Pydantic Model & System Prompt
# --------------------------------------------------------------------------------
class ConversationResponse(BaseModel):
    # ANALYZE
    user_intent: Literal["greeting", "complaint", "storytelling", "question", "farewell"]
    thought: str = Field(..., description="Brief internal reasoning about the user's intent and how to reply.")

    # CATEGORIZATION
    gesture: Literal["nod", "wave", "shake_head", "idle", "point"]
    emotion: Literal["neutral", "happy", "sad", "excited", "confused"]
    
    # DRAFT & REFINE
    message      : str = Field(..., description="The spoken response to the user.")
    
    #critique     : str = Field(..., description="Check if the draft is simple, empathetic, and concise.")
    #final_message: str = Field(..., description="The revised message based on the critique.")
    #conversation_state: Literal["listening", "processing", "closing", "clarifying"]
    

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

# Parts of the output format that are currently commented out
REMOVED = """
- "critique": Check if the draft is simple, empathetic, and concise.
- "final_message": The revised, final spoken text.
- "conversation_state": [listening, processing, closing, clarifying]
"""

# --------------------------------------------------------------------------------
# Get a response from the LLM
# --------------------------------------------------------------------------------
def get_response(client, user_prompt):
    print(f"{CYAN}Sending request...{RESET}")
    
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content":   user_prompt}
            ],
            response_model = ConversationResponse,
            temperature    = 0.5,
            max_tokens     = 512,
        )
        t1 = time.time()
        duration = t1 - t0
        
        # Print model response
        print(f"{CYAN}--- MODEL RESPONSE ({duration:.2f}s) ---{RESET}")
        print(f"{YELLOW}User:        {user_prompt}")
        print(f"{GREEN}Intent:     {RESET} {response.user_intent}")
        print(f"{GREEN}Thought:    {RESET} {response.thought}")
        #print(f"{GREEN}State:      {RESET} {response.conversation_state}")
        print(f"{GREEN}Emotion:    {RESET} {response.emotion}")
        print(f"{GREEN}Gesture:    {RESET} {response.gesture}")
        print(f"{GREEN}Message:    {RESET} {response.message}")
        #print(f"{GREEN}Critique:   {RESET} {response.critique}")
        #print(f"{GREEN}Final:      {RESET} {response.final_message}")
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
    client = instructor.from_openai(
        OpenAI(
            base_url = llm_url, 
            api_key  = llm_key,
            timeout  = 20.0, 
        ),
        mode=instructor.Mode.JSON
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
EOF

# ================================================================================
# 4. Create Dockerfile
# ================================================================================
log_step "4" "Generating Dockerfile"
cat <<EOF > Dockerfile
FROM python:3.11-slim
WORKDIR /app

# Force python to print
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir instructor openai pydantic httpx
COPY main.py .
CMD ["python", "main.py"]
EOF

# ================================================================================
# 5. Build Image
# ================================================================================
log_step "5" "Building Docker Image"
echo -e "${YELLOW}Building... ${NC}"

if sudo docker build -t struct-bot-image . ; then
    echo -e "${GREEN}Build Successful!${NC}"
else
    echo -e "${RED}Build Failed!.${NC}"
    # Re-run visibly if silent build failed
    sudo docker build -t struct-bot-image .
    exit 1
fi

# ================================================================================
# 6. Run Container
# ================================================================================
log_step "6" "Running Container"
echo -e "${YELLOW}Structured Generation${NC}"

sudo docker run --rm --network="host" \
  -e LLM_URL="$TARGET_URL" \
  -e LLM_KEY="$TARGET_KEY" \
  struct-bot-image

# ================================================================================
# 7. Cleanup
# ================================================================================
log_step "7" "Cleaning up temporary files"
cd ..
sudo rm -rf struct_bot_build
echo -e "${GREEN}Cleanup complete. Execution finished.${NC}"
