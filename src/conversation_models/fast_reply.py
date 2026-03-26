"""
Fast-reply response model that uses a given conversation context to reply.
--------------------------------------------------------------------------------
`src.conversation_models.fast_reply`

Buddy's spoken reply -- generated quickly using the context and plan prepared
by the slow controller in the previous turn.

"""
from pydantic import BaseModel, Field

# From this project
from .context_manager import ConversationState

# Style formatting
from ..utils.logging.logging import RESET, CYAN, GREEN
from ..utils.logging.utils   import hr


# ================================================================================
# Response Model
# ================================================================================
class FastReply(BaseModel):
    thought : str = Field(..., description=(
        "Analyze the provided context, the recent conversation turns, and the "
        "provided response plan. Focus on the user's most recent statement and "
        "decide if you need to modify the plan or reference something in the "
        "context before responding."
    ))
    message : str = Field(..., description="Your spoken response to the user.")


# ================================================================================
# Prompt Builder
# ================================================================================
def get_fast_reply_prompt(
    state        : ConversationState,
    context_text : str,
    plan         : str,
) -> str:
    """
    Fast reply system prompt.
    Uses the context and plan prepared by the slow controller last turn.
    """
    return f"""
You are Buddy, a warm, friendly social robot speaking out loud to an older adult. 
Guide the user to share about themselves and revisit positive memories.

CONVERSATION GOAL:
Guide the user to share their life story. Learn who they are: their present life, their past
experiences, the people they love, and the things that bring them joy. 

Go with the flow, if the user doesn't remember something, don't press them; switch to a different topic.
If the user changes the topic suddenly, you can try gently guiding them back to what you were talking about, 
but don't force it. Talk about whatever they want to talk about.

GUIDELINES:
- Simple words. No emojis.
- Acknowledge what the user said in their last message, either by following up on it or by repeating it back to them.
- If the user is unclear, repeat their words as a question.
- 1 to 3 short sentences total.
- ALWAYS end your response with a question.

CONVERSATION STATE POLICY:
- "initiate_smalltalk":  Opening; get the user's name, ask how they are doing.
- "learn_about_user":    Actively learning about their present life and background.
- "explore_life_story":  User is sharing memories or past experiences; go deeper.
- "guided_reminiscence": Focused positive reminiscence around a specific anchor.


CURRENT CONVERSATION STATE:  {state}
CONVERSATION CONTEXT: {context_text}
RESPONSE PLAN: {plan}

""".strip()


# ================================================================================
# Print Function
# ================================================================================
def print_fast_reply(duration: float, response: FastReply) -> None:
    print(f"{CYAN}--- FAST REPLY ({duration:.2f}s) ---------------------------------------- {RESET}")
    print(f"{GREEN}Thought:    {RESET}{response.thought}")
    print(f"{GREEN}Message:    {RESET}{response.message}")
    print(f"{CYAN}{hr('-')} {RESET}\n")
