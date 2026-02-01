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
import json

from pydantic import BaseModel, Field
from typing   import Literal
from openai   import OpenAI

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# Colors
GREY     = '\033[90m'
RED      = '\033[91m'
GREEN    = '\033[92m'
YELLOW   = '\033[93m'
BLUE     = '\033[94m'
MAGENTA  = '\033[95m'
CYAN     = '\033[96m'
WHITE    = '\033[97m'
RESET    = '\033[0m'

LLM_URL = os.getenv("LLM_URL", "http://localhost:8000/v1")
LLM_KEY = os.getenv("LLM_KEY", "TOKEN")

# MODEL SELECTION
# Options: phi3-buddy | phi3.5-mini | qwen2.5-3b | qwen2.5-3b-speculative | qwen2.5-0.5b
MODEL = "phi3.5-mini" 

# ================================================================================
# [ROBOT] Pydantic Model & System Prompt
# ================================================================================
class ConversationResponse(BaseModel):
    # Analyze the user's message
    user_intent: Literal["greeting", "complaint", "storytelling", "question", "farewell"]
    thought: str = Field(..., description="Brief internal reasoning about the user's intent and how to reply.")

    # Do this before the message, so it decides the state and then drafts the response to say with that in mind
    conversation_state: Literal["start_conversation", "initiate_smalltalk", "explore_user_interests", "initiate_memory_activity", "discuss_memory_activity_topic"]
    
    # Draft response
    message: str = Field(..., description="The spoken response to the user.")

# Robot System Prompt TODO: Change this to pull from the .json file
ROBOT_SYSTEM_PROMPT = """
ROLE: You are Buddy, a warm, friendly robot built by Indiana University. You love listening to stories about the past.

GUIDELINES:
1. STYLE: Use simple words. Max 2 short sentences. NO emojis.
2. EMPATHY: Validate feelings first.
3. CLARITY: If the user is unclear, repeat their words as a question.
4. FLOW: Engage with the user. ALWAYS end with a simple follow-up question.
5. SAFETY: Do NOT give medical advice. The user CANNOT see your internal JSON or code.

CONVERSATION_STATE LOGIC:
- "start_conversation": Used at the beginning of interaction, when there is no previous chat history with the user.
- "initiate_smalltalk": Comes after the initial introduction with the user where the user shares their name.
- "explore_user_interests": Comes after the "initiate_smalltalk" state when the user naturally shares something they are interested in.
- "initiate_memory_activity": After exploring the user's personal interest, natually transition toward starting a memory activity where you discuss a memorable event from the user's past.
- "discuss_memory_activity_topic": After the user selects a particular topic for the memory activity, start discuss the topic with the user.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object.
"""

# Print the robots turn
def print_robot_turn(duration, user_message, response):
    print(f"{CYAN} --- ROBOT RESPONSE ({duration:.2f}s) ------------------------- {RESET}")
    print(f"{YELLOW} User:        {user_message}")
    print(f"{GREEN} Intent:     {RESET} {response.user_intent}")
    print(f"{GREEN} Thought:    {RESET} {response.thought}")
    print(f"{GREEN} State:      {RESET} {response.conversation_state}")
    print(f"{GREEN} Message:    {RESET} {response.message}")
    print(f"{CYAN} -------------------------------------------------------------- {RESET}\n")

# ================================================================================
# [USER] Pydantic Model & System Prompt
# ================================================================================
# TODO: Maybe could add adaptions to make it more like their thought during the demos? (e.g., trying to figure out the robots limits)
class UserConversationResponse(BaseModel):
    #thought: str = Field(..., description="Brief internal reasoning about the robot's intent and how to reply.")
    message: str = Field(..., description="The spoken response to the robot.")

# User System Prompt (Grandma)
USER_SYSTEM_PROMPT = """
ROLE: You are Martha, an 82-year-old woman living with early-stage dementia.

CONTEXT: You are participating in a study to test a conversational robot helper named Buddy.

GUIDELINES:
- Use simple words. Max 2 short sentences. NO emojis.
- You are sometimes confused about what day it is.
- You love talking about your garden.
- You answer simply and conversationaly.
"""

# Print the user's turn
def print_user_turn(duration, robot_message, response):
    print(f"{MAGENTA} --- USER RESPONSE ({duration:.2f}s) -------------------------- {RESET}")
    print(f"{YELLOW} Robot:       {robot_message}")
    print(f"{GREEN} Message:    {RESET} {response.message}")
    print(f"{MAGENTA} -------------------------------------------------------------- {RESET}\n")



# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
# Update both histories whenever a new message comes in
def sync_histories(history_robot, history_user, response_data, speaker_role: Literal["ROBOT", "USER"]):
    """
    - Robot stores its own JSON but only hears plain text from User.
    - User only stores plain text for both its own and the robots responses
    """
    if   speaker_role == "ROBOT": sync_history_ROBOT(history_robot, history_user, response_data)
    elif speaker_role == "USER" : sync_history_USER (history_robot, history_user, response_data)

# ROBOT spoke (user hears plain text only; robot saves its internal processes) 
# TODO: Remove the "intent" and "thought" from the JSON, just keep the conversation state and message
def sync_history_ROBOT(history_robot, history_user, response_data):
    history_robot.append({"role": "assistant", "content": response_data.model_dump_json()})
    history_user .append({"role": "assistant", "content": response_data.message})

# USER spoke (both agents just hear plain text)
def sync_history_USER(history_robot, history_user, response_data):
    history_robot.append({"role": "user", "content": response_data.message})
    history_user .append({"role": "user", "content": response_data.message})


# ================================================================================
# Loop Function
# ================================================================================
def run_simulation(turns=3):
    print(f"{YELLOW}Starting Simulation: Buddy vs. Grandma (Structured){RESET}\n")

    # --------------------------------------------------------------------------------
    # 1) Initialize Client & Histories
    # --------------------------------------------------------------------------------
    # Client in instructor mode
    client = instructor.from_openai(
        OpenAI(base_url=LLM_URL, api_key=LLM_KEY, timeout=20.0),
        mode=instructor.Mode.JSON
    )

    # Separate histories (user doesn't know about robots thought processes & vice versa)
    history_robot = [{"role": "system", "content": ROBOT_SYSTEM_PROMPT}]
    history_user  = [{"role": "system", "content":  USER_SYSTEM_PROMPT}]

    # --------------------------------------------------------------------------------
    # 2) Begin the Conversation (robot goes first)
    # --------------------------------------------------------------------------------
    # Manually send the greeting so the user has something to reply to
    start_message = "Good morning! My name is Buddy. It is nice to meet you."
    
    print(f"{CYAN}BUDDY (Start):{RESET} {start_message}\n")
    
    # Sync Histories
    history_robot.append({"role": "assistant", "content": json.dumps({"user_intent": "greeting", "thought": "intro", "conversation_state": "start_conversation", "message": start_message,})})
    history_user .append({"role": "assistant", "content": start_message})

    # ================================================================================
    # 3) Main Loop
    # ================================================================================
    last_robot_message = start_message

    for i in range(turns):
        print(f"{WHITE}================ Turn {i+1} ================ {RESET}\n")

        # --------------------------------------------------------------------------------
        # a) USER Speaks
        # --------------------------------------------------------------------------------
        print(f"{MAGENTA}[USER] Sending request...{RESET}")
        t0 = time.time()
        
        # User thinks based on Robot's last message
        user_response = client.chat.completions.create(
            model          = MODEL,
            messages       = history_user,
            response_model = UserConversationResponse, # Structured
            temperature    = 0.5, 
            max_tokens     = 512,
        )
        
        t1 = time.time()
        print_user_turn(t1-t0, last_robot_message, user_response)

        # Sync histories
        sync_histories(history_robot, history_user, user_response, speaker_role="USER")

        # --------------------------------------------------------------------------------
        # b) ROBOT Speaks
        # --------------------------------------------------------------------------------
        print(f"{CYAN}[ROBOT] Sending request...{RESET}")
        t0 = time.time()
        
        # Robot thinks based on User's message
        robot_response = client.chat.completions.create(
            model          = MODEL,
            messages       = history_robot,
            response_model = ConversationResponse, # Structured
            temperature    = 0.1,                  # Lower temperature for consistency
            max_tokens     = 512
        )
        
        t1 = time.time()
        print_robot_turn(t1-t0, user_response.message, robot_response)

        # Sync histories
        sync_histories(history_robot, history_user, robot_response, speaker_role="ROBOT")

        # Update tracking variable for next print
        last_robot_message = robot_response.message

        # Sleep for readability
        time.sleep(1)

# --------------------------------------------------------------------------------
# Server Calls
# --------------------------------------------------------------------------------
print(f"{YELLOW}Attempting connection to: {LLM_URL}...{RESET}")
print(f"{YELLOW}Model endpoint: {MODEL} {RESET}\n")

try:
    run_simulation(turns=3)

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
echo -e "${YELLOW}Looped Conversation w Structured Generation${NC}"

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
