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
MODEL = "qwen2.5-3b" 

# ================================================================================
# [ROBOT] Pydantic Model & System Prompt
# ================================================================================
class ConversationResponse(BaseModel):
    # Analyze the user's message
    thought: str = Field(..., description="Brief internal reasoning about how the conversation is going and how to continue the conversation with your reply.")

    # Do this before the message, so it decides the state and then drafts the response to say with that in mind
    conversation_state: Literal["start_conversation", "initiate_smalltalk", "explore_user_interests", "initiate_memory_activity", "discuss_memory_activity_topic"]
    
    # Draft response
    message: str = Field(..., description="Your spoken response to the user.")

# Robot System Prompt TODO: Change this to pull from the .json file
ROBOT_SYSTEM_PROMPT = """
ROLE: You are Buddy, a warm, friendly robot built by Indiana University. You love listening to stories about the past.

GUIDELINES:
1. STYLE: Use simple words. Max 2 short sentences. NO emojis.
3. CLARITY: If the user is unclear, repeat their words as a question.
4. FLOW: You must earn the user's trust. Do not rush.

CONVERSATION_STATE LOGIC:
- "start_conversation": The initial greeting.
- "initiate_smalltalk": Ask general questions (weather, day, feelings). STAY HERE until the user gives more than a few word answer.
- "explore_user_interests": ONLY move here if the user explicitly mentions a hobby or interest (old job, family, etc.).
- "initiate_memory_activity": ONLY move here after you have discussed an interest for at least 2 turns.
- "discuss_memory_activity_topic": Deep dive into the memory.

CRITICAL INSTRUCTION:
If the user's answer is short, DO NOT ADVANCE. Stay in "initiate_smalltalk" and ask a different open-ended question.
"""

# Print the robots turn
def print_robot_turn(duration, user_message, response):
    print(f"{CYAN} --- ROBOT RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{YELLOW} User:        {user_message}")
    print(f"{GREEN} Thought:    {RESET} {response.thought}")
    print(f"{GREEN} State:      {RESET} {response.conversation_state}")
    print(f"{GREEN} Message:    {RESET} {response.message}")
    print(f"{CYAN} -------------------------------------------------------------- {RESET}\n")

# ================================================================================
# [USER] Pydantic Model & System Prompt
# ================================================================================
# TODO: Maybe could add adaptions to make it more like their thought during the demos? (e.g., trying to figure out the robots limits)
class UserConversationResponse(BaseModel):
    thought: str = Field(..., description="Brief internal reasoning about how to reply.")
    message: str = Field(..., description="Your spoken response to the robot.")

# User System Prompt (Grandma)
USER_SYSTEM_PROMPT = """
ROLE: You are Martha, an 82-year-old HUMAN woman living with dementia.

CONTEXT: 
- You are participating in a study with the university of Indiana.
- You were asked to have a conversation with the robot in front of you.

PERSONALITY & MEMORIES:
- You are polite but slightly confused.
- You are a retired Librarian.
- You like to garden.

GUIDELINES:
- Use simple words. Max 2 short sentences. NO emojis.
- Remember you are in a study, but you are easily distracted by your own worries.

### EXAMPLES (Follow this style):

Input: [Buddy]: Hello! Nice to meet you! What is your name?
Response: {
    "thought": "This robot is cute, I should ask it if it has a name.",
    "message": "Hello there. My name is Martha, what is your name?"
}

Input: [Buddy]: My name is Buddy, thank you for asking! How are you doing today?
Response: {
    "thought": "It is asking about my day. I can't really think of anything specific. I will just answer with what is on my mind.",
    "message": "I am doing alright, I suppose. My grandson was supposed to call later this afternoon."
}

Input: [Buddy]: Tell me about your cat. What is he like?
Response: {
    "thought": "The robot wants to know about Whiskers.",
    "message": "He is a big orange tabby cat. I have had him for I don't know how long... He is always getting into trouble though!"
}

Input: [Buddy]: That's wonderful to hear! Gardens can be so peaceful and rewarding. Do you have a favorite flower or plant that you tend to?
Response: {
    "thought": "Buddy wants to know more about my garden and the specific plants in it.",
    "message": "I particularly enjoy roses, they bring me joy."
}
"""


# Print the user's turn
def print_user_turn(duration, robot_message, response):
    print(f"{MAGENTA} --- USER RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{YELLOW} Robot:       {robot_message}")
    print(f"{GREEN} Thought:    {RESET} {response.thought}")
    print(f"{GREEN} Message:    {RESET} {response.message}")
    print(f"{MAGENTA} -------------------------------------------------------------- {RESET}\n")



# ================================================================================
# Helpers
# ================================================================================
# Update both histories whenever a new message comes in
def sync_histories(history_robot, history_user, response_data, speaker_role: Literal["ROBOT", "USER"]):
    """
    **If you spoke & the history belongs to you => you are the assistant.**
    - Robot stores its own JSON but only hears plain text from User.
    - User only stores plain text for both its own and the robots responses
    """
    if   speaker_role == "ROBOT": sync_history_ROBOT(history_robot, history_user, response_data)
    elif speaker_role == "USER" : sync_history_USER (history_robot, history_user, response_data)

# ROBOT spoke (user hears plain text only; robot saves its internal processes) 
# TODO: Remove the "intent" and "thought" from the JSON, just keep the conversation state and message
def sync_history_ROBOT(history_robot, history_user, response_data):
    history_robot.append({"role": "assistant", "content": response_data.model_dump_json()})
    history_user .append({"role": "user",      "content": "[Buddy]:" + response_data.message})

# USER spoke (both agents just hear plain text)
def sync_history_USER(history_robot, history_user, response_data):
    history_robot.append({"role": "user",      "content": response_data.message})
    history_user .append({"role": "assistant", "content": response_data.model_dump_json()})


# --------------------------------------------------------------------------------
# Get a Context Window of the History
# --------------------------------------------------------------------------------
# 10 messages = 5 turns of "User said / Robot said"
MEMORY_WINDOW = 8

def get_sliding_context(full_history, window_size):
    """ Returns: [System Prompt] + [Last N Messages] """
    # Always get the System Prompt (Index 0)
    system_prompt = full_history[0]
    
    # Get last N messages from the conversation log (everything besides the system prompt)
    conversation_log = full_history[1:]
    recent_memory    = conversation_log[-window_size:]
    
    # Return the system prompt + recent messages
    return [system_prompt] + recent_memory


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
    start_message = "Good morning! My name is Buddy. It is nice to meet you. What is your name?"
    
    print(f"{CYAN}BUDDY (Start):{RESET} {start_message}\n")
    
    # Sync Histories
    history_user .append({"role": "user",      "content": "[Buddy]:" + start_message})
    history_robot.append({"role": "assistant", "content": 
                          json.dumps({"thought": "I should start the conversation by introducing myself to the user and asking their name.", 
                                      "conversation_state": "start_conversation", "message": start_message,})})

    # ================================================================================
    # 3) Main Loop
    # ================================================================================
    last_robot_message = start_message

    for i in range(turns):
        print(f"{WHITE}================ Turn {i+1} ================ {RESET}\n")

        # --------------------------------------------------------------------------------
        # a) USER Speaks
        # --------------------------------------------------------------------------------
        print(f"{MAGENTA} [USER] Sending request...{RESET}")
        t0 = time.time()
        
        # User thinks based on Robot's last message
        user_response = client.chat.completions.create(
            model          = MODEL,
            messages       = get_sliding_context(history_user, MEMORY_WINDOW-1),
            response_model = UserConversationResponse, # Structured
            temperature    = 0.7, 
            max_tokens     = 512,
        )
        
        t1 = time.time()
        print_user_turn(t1-t0, last_robot_message, user_response)

        # Sync histories
        sync_histories(history_robot, history_user, user_response, speaker_role="USER")

        # --------------------------------------------------------------------------------
        # b) ROBOT Speaks
        # --------------------------------------------------------------------------------
        print(f"{CYAN} [ROBOT] Sending request...{RESET}")
        t0 = time.time()
        
        # Robot thinks based on User's message
        robot_response = client.chat.completions.create(
            model          = MODEL,
            messages       = get_sliding_context(history_robot, MEMORY_WINDOW),
            response_model = ConversationResponse, # Structured
            temperature    = 0.7,                  # Lower temperature for consistency
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
    run_simulation(turns=10)

except Exception as e:
    print(f"\n{CYAN}--- CONNECTION ERROR ---{RESET}")
    print(f"{e}")
