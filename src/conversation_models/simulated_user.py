"""
Simulate user responses for a "realistic" back-and-forth conversation.
--------------------------------------------------------------------------------
`src.conversation_models.simulated_user`

This simulates a character, Martha, meant to possibly exhibit symptoms of
dementia.

The response model can either be just the message, or also include a 'thought'
field. 

"""
from pydantic import BaseModel, Field

# From this project
from ..utils.logging.logging import RESET, MAGENTA, YELLOW, GREEN
from ..utils.logging.utils   import hr


# ================================================================================
# Pydantic Models
# ================================================================================
# Include visible reasoning (thought field)
class UserConversationResponse(BaseModel):
    thought: str = Field(..., description="Brief internal reasoning about how to reply.")
    message: str = Field(..., description="Your spoken response to the robot.")

# Message only
class UserMessageOnlyResponse(BaseModel):
    message: str = Field(..., description="Your spoken response to the robot.")


# ================================================================================
# Print Functions
# ================================================================================
# Include visible reasoning (thought field)
def print_user_turn(duration: float, response: UserConversationResponse):
    print(f"{MAGENTA}--- USER RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{GREEN  }Thought: {RESET} {response.thought}")
    print(f"{GREEN  }Message: {RESET} {response.message}")
    print(f"{MAGENTA}{hr('-')}{RESET}\n")

# Message only
def print_user_turn_v2(duration: float, last_robot_message: str, user_message: str):
    print(f"{MAGENTA}--- USER RESPONSE ({duration:.2f}s) ----------------------------------- {RESET}")
    print(f"{GREEN  }Message: {RESET}{user_message}")
    print(f"{MAGENTA}{hr('-')}{RESET}\n")



# --------------------------------------------------------------------------------
# User System Prompt (Grandma)
# --------------------------------------------------------------------------------
USER_SYSTEM_PROMPT = """
ROLE: You are Martha, an 82-year-old HUMAN woman living with dementia.

CONTEXT: 
- You are participating in a study with Indiana University.
- You were asked to have a conversation with the robot in front of you.

PERSONALITY & MEMORIES:
- You are polite but slightly confused.
- You are a retired Librarian.
- You like to garden.

GUIDELINES:
- Use simple words. Keep responses to around 4-5 short sentences. NO emojis.
- Remember you are in a study, but you are easily distracted by your own worries.
- You can just talk about yourself or whatever is on your mind.

"""

# Taking the examples out temporarily
"""
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


# ================================================================================
# Shorter Prompt 
# ================================================================================
USER_SYSTEM_PROMPT_V2 = """
ROLE: You are Martha, an 82-year-old HUMAN woman living with dementia.

CONTEXT:
- You are participating in a study with Indiana University.
- You are having a conversation with the robot in front of you.

PERSONALITY & MEMORIES:
- Use simple words. NO emojis.
- You are a retired librarian.
- You like to garden.


### EXAMPLES (Follow this style)

Input: [Buddy]: Hello! Nice to meet you! What is your name?
Response: {"message": "Hi! My name is Martha, what is your name?"}

Input: [Buddy]: How do you feel when you see snow falling outside?
Response: {"message": "When you see snow falling, it's a very nice feeling. When I look at it piled up everywhere, I think blah."}

""".strip()

