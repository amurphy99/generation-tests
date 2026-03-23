import os
import time
import json
import instructor

from dataclasses import dataclass
from typing import Optional, Type, Literal, List, Dict, Any

from pydantic import BaseModel, Field
from openai import OpenAI

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
# Colors
GREY     = "\033[90m"
RED      = "\033[91m"
GREEN    = "\033[92m"
YELLOW   = "\033[93m"
BLUE     = "\033[94m"
MAGENTA  = "\033[95m"
CYAN     = "\033[96m"
WHITE    = "\033[97m"
RESET    = "\033[0m"

# Extended Colors
ORANGE       = "\033[38;5;208m"
PINK         = "\033[38;5;205m"
TEAL         = "\033[38;5;51m"
GOLD         = "\033[38;5;220m"
PURPLE       = "\033[38;5;93m"
LIME         = "\033[38;5;154m"
SKY_BLUE     = "\033[38;5;39m"

# Standard (Dark) Colors
BLACK_STD    = "\033[30m"
RED_STD      = "\033[31m"
GREEN_STD    = "\033[32m"
YELLOW_STD   = "\033[33m"
BLUE_STD     = "\033[34m"
MAGENTA_STD  = "\033[35m"
CYAN_STD     = "\033[36m"
WHITE_STD    = "\033[37m"

# Formatting
BOLD         = "\033[1m"
DIM          = "\033[2m"
ITALIC       = "\033[3m"
UNDERLINE    = "\033[4m"
BLINK        = "\033[5m"
REVERSE      = "\033[7m"


# LLM access
LLM_URL = os.getenv("LLM_URL", "http://localhost:8000/v1")
LLM_KEY = os.getenv("LLM_KEY", "TOKEN")

# MODEL SELECTION
# Options: phi3-buddy | phi3.5-mini | qwen2.5-3b | qwen2.5-3b-speculative | qwen2.5-0.5b
MODEL = "qwen2.5-3b" 

# ================================================================================
# Agent Config (dataclass)
# ================================================================================
@dataclass(frozen=True)
class AgentConfig:
    name           : str
    model          : str
    temperature    : float
    max_tokens     : int
    window         : int
    response_model : Optional[Type[BaseModel]] = None  # None => raw text


# ================================================================================
# Pydantic Models
# ================================================================================
# Enum for the conversation states
ConversationState = Literal[
    "initiate_smalltalk",
    "explore_user_interests",
    "initiate_memory_activity",
    "discuss_memory_activity_topic",
]

# --------------------------------------------------------------------------------
# Response Models
# --------------------------------------------------------------------------------
# Simulated user
class UserConversationResponse(BaseModel):
    message: str = Field(..., description="Your spoken response to the robot.")

# Fast robot replies
class RobotFastReply(BaseModel):
    message: str = Field(..., description="Your spoken response to the user.")

# Slow, "thinking" robot controller
class RobotSlowUpdate(BaseModel):
    state_reason       : str = Field(..., description="Justification for state change, citing user message.")
    conversation_state : ConversationState
    evidence           : str = Field(..., description="Compressed fragments extracted from the USER message only. Use short phrase chunks separated by ...")
    context_delta      : List[str] = Field(..., description=(
            "A list of compact context update operations to apply to the persistent ConversationContext. "
            "Use 'key=value' to set scalar fields and 'key+=value' to append to list fields. "
            "Include only stable, useful facts; omit if no updates are needed."
        ),
    )
    tentative_plan     : str = Field(..., description="A brief suggestion to the assistant about how they might advance the conversation depending on what the user's next reply could be.")


# ================================================================================
# System Prompts
# ================================================================================
# Simulated User Prompt
USER_SYSTEM_PROMPT = """
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

# --------------------------------------------------------------------------------
# Prompt Builders (fast and slow robot models use dynamic system prompts)
# --------------------------------------------------------------------------------
# [FAST] Resonse Model Prompt
def get_robot_fast_prompt(state: ConversationState, context_text: str, plan: str) -> str:
    """
    Fast model prompt:
    - keep short
    - depends on controller-provided state + retained context
    """
    return f"""
You are Buddy, a warm, friendly social robot speaking out loud to an older adult.

Hard constraints:
- Simple words. No emojis.
- 1 to 3 short sentences total.
- ALWAYS end your responses with a question.


CONVERSATION STATE: {state}
CONVERSATION CONTEXT: {context_text}
RESPONSE PLAN: {plan}


### EXAMPLES (Follow this style)

Input: Good morning! My name is Robert.
Response: {{"message": "Nice to meet you, Robert. What's something that's on your bucket list?"}}

Input: I like spending time outside, and I've been trying to walk more this year.
Response: {{"message": "That sounds great, Robert. Where do you like to go for walks?"}}

Input: I used to travel a lot for work, and I still miss seeing new places.
Response: {{"message": "That sounds interesting, Patrick. What place do you miss the most?"}}

""".strip()

# [SLOW] Thinking Robot Prompt 
# Updates conversation_state if needed, updates context if new information gained, and gives additional instructions
def get_robot_slow_prompt(current_state: ConversationState, context_json: str) -> str:
    """
    Slow Controller Prompt
    -----------------------
    Keeps `conversation_state` and `context` "sticky-by-default" (e.g. don't change 
    unless there is good reason).

    Context changes only via "deltas". These are operations to the fields of the 
    context JSON object, and should keep tokens down.

    It also gives a `tentative_plan` to the fast model. Tentative because we will
    be one turn behind the fast models knowledge of the user's utterances. 
    """
    return f"""
You are Buddy's CONTROLLER. You do NOT speak to the user.
Guide Buddy toward a discussion connecting the user's current hobby/interest to a familiar past topic/era/event, then prompt for personal memories (storytelling).


STATE POLICY:
- Default is NO CHANGE: keep the same conversation_state unless the last user message gives a clear reason to change.
- If you change conversation_state, state_reason must cite what triggered it from the last user message.


CONTEXT POLICY & SELECTIVITY RULES (STRICT):
- Default is NO CHANGE.
- ONLY add a context_delta if it is NEW information not already present in the current context.
- Do NOT repeat existing or similar items.
- Each list item must be ONE fact (no commas, no "and").
- If it is longer than three words, it CANNOT be in the user_profile.
- Put detailed titles/names into key_entities instead.

DELTA FORMAT:
Each item in context_delta must be a single string in ONE of these forms:
- "<field>=<value>"    (set scalar field)
- "<field>+=<value>"   (append to list field)

ALLOWED DELTA FIELDS:
user_name, user_profile, key_entities, open_threads, last_focus


Current conversation_state: 
{current_state}

Current context (JSON):
{context_json}


### EXAMPLES

Input: My name is Robert. Nice to meet you too. I'm a retired teacher.
Response: {{"state_reason": "Introductions completed; new background detail given.", "conversation_state": "initiate_smalltalk", "evidence": "name is Robert ... retired teacher", "context_delta": ["user_name=Robert", "user_profile+=retired teacher"], "tentative_plan": "Ask what grade or subject he taught."}}

Input: I'm doing okay. My grandson is supposed to call later today.
Response: {{"state_reason": "User shared mood and a near-term family event.", "conversation_state": "initiate_smalltalk", "evidence": "doing okay ... grandson will call later today", "context_delta": ["key_entities+=grandson", "open_threads+=Ask about the grandson's call", "last_focus=family"], "tentative_plan": "Ask what she and her grandson usually talk about."}}

Input: I don't really have hobbies. I mostly just watch TV.
Response: {{"state_reason": "User gave a simple routine-based interest.", "conversation_state": "initiate_smalltalk", "evidence": "no hobbies ... watches TV", "context_delta": ["user_profile+=watches TV", "last_focus=daily routine"], "tentative_plan": "Ask what shows she likes or watched recently."}}

Input: I used to work as a librarian. I miss the quiet sometimes.
Response: {{"state_reason": "Past work and preference can be explored as an interest.", "conversation_state": "explore_user_interests", "evidence": "used to be librarian ... misses the quiet", "context_delta": ["user_profile+=retired librarian", "user_profile+=enjoys quiet", "last_focus=work history"], "tentative_plan": "Ask what she liked most about the library."}}

Input: I grew up in Ohio, near Cleveland.
Response: {{"state_reason": "User shared a childhood location worth exploring.", "conversation_state": "explore_user_interests", "evidence": "grew up in Ohio ... near Cleveland", "context_delta": ["key_entities+=Cleveland, Ohio", "user_profile+=grew up in Ohio", "last_focus=childhood"], "tentative_plan": "Ask what her neighborhood was like growing up."}}

Input: We used to have a big garden when I was a kid.
Response: {{"state_reason": "User linked an interest to a past memory cue.", "conversation_state": "initiate_memory_activity", "evidence": "big garden as a kid ... childhood gardening", "context_delta": ["user_profile+=had childhood garden", "last_focus=childhood memories"], "tentative_plan": "Ask what they grew and who gardened with them."}}

""".strip()

# ================================================================================
# Helpers
# ================================================================================
# History Management
def get_sliding_context(system_prompt_text: str, full_history: list, window_size: int) -> list:
    """ Builds a `history` with a given system prompt + last N messages. """
    system_msg = {"role": "system", "content": system_prompt_text}
    recent     = full_history[-window_size:] if window_size > 0 else []
    return [system_msg] + recent

# Helper to get a response from an agent
def run_agent(
    client_raw    : OpenAI,
    client_structured,
    agent         : AgentConfig,
    system_prompt : str,
    history       : list,
):
    """
    Runs the OpenAI web API call for all agent configurations.
    - If agent.response_model is None => raw text
    - Else => instructor/pydantic enforced structured output
    """
    # Prepare a history to provide the model with
    messages = get_sliding_context(system_prompt, history, agent.window)

    # For regular, non-structure generation
    if agent.response_model is None:
        completion = client_raw.chat.completions.create(
            model       = agent.model,
            messages    = messages,
            temperature = agent.temperature,
            max_tokens  = agent.max_tokens,
        )
        return completion.choices[0].message.content.strip()

    # Structure generation query
    return client_structured.chat.completions.create(
        model          = agent.model,
        messages       = messages,
        response_model = agent.response_model,
        temperature    = agent.temperature,
        max_tokens     = agent.max_tokens,
    )


# ================================================================================
# Long-term Context Storage (persistent across turns)
# ================================================================================
# Pydantic Conversation Context
class ConversationContext(BaseModel):
    user_name         : Optional[str] = None
    user_profile      : List[str] = Field(default_factory=list) # Stable facts (ex: retired librarian, likes gardening)
    key_entities      : List[str] = Field(default_factory=list) # People/pets/places mentioned
    open_threads      : List[str] = Field(default_factory=list) # Questions to return to later
    last_focus        : Optional[str] = None                    # What we're talking about now
    #memory_topic      : Optional[str] = None
    turns_in_interest : int = 0

# Default init
def init_context_store() -> ConversationContext:
    return ConversationContext()

# --------------------------------------------------------------------------------
# Apply generated deltas to the locally stored ConversationContext
# --------------------------------------------------------------------------------
def apply_context_delta(ctx: ConversationContext, deltas: List[str]) -> ConversationContext:
    """
    Accepts strict deltas:
      - "user_name=Martha"
      - "user_profile+=likes gardening"
    """
    for raw in deltas:
        s = (raw or "").strip()
        if not s:  continue

        # Normalize common prefixes like "key+=" or "key="
        lowered = s.lower()
        if   lowered.startswith("key+="): s = s[5:].lstrip()
        elif lowered.startswith("key=" ): s = s[4:].lstrip()

        # Normalize ":" to "=" if model used a colon 
        if ":" in s and "=" not in s and "+=" not in s: s = s.replace(":", "=", 1)

        # Parse the operator
        op = None
        if   "+=" in s: key, val = s.split("+=", 1); op = "+="
        elif  "=" in s: key, val = s.split( "=", 1); op =  "="
        else: continue

        key = key.strip()
        val = val.strip()
        if not key or not hasattr(ctx, key): continue

        current = getattr(ctx, key)

        # If they used "+=" on a scalar field, treat it as "=" 
        if op == "+=" and not isinstance(current, list):  op = "="

        if op == "+=":
            if isinstance(current, list) and val and val not in current: current.append(val)

        elif op == "=":
            # Don't allow scalar set on list fields
            if isinstance(current, list):  continue

            if isinstance(current, int):
                try: setattr(ctx, key, int(val))
                except ValueError: continue
            else: setattr(ctx, key, val if val != "" else None)

    return ctx



# Dump the context 
def context_to_json(ctx: ConversationContext) -> str:
    return ctx.model_dump_json()

# Convert the JSON context to something for the fast response model to use
def render_context_for_fast(ctx: ConversationContext) -> str:
    """
    Turn persistent context into 1-3 short sentences for the fast prompt.
    Trying to be efficient with tokens and only include relevant information.
    """
    bits: List[str] = []

    # User name
    if ctx.user_name: bits.append(f"User name is {ctx.user_name}.")

    # Keep only the most important/stable facts (if doing so, ctx.user_profile[:2])
    if ctx.user_profile:
        top_profile = "; ".join(ctx.user_profile)
        bits.append(f"About user: {top_profile}.")

    # People/pets/places etc. (ctx.key_entities[:2])
    if ctx.key_entities:   
        top_entities = ", ".join(ctx.key_entities)
        bits.append(f"Key entities: {top_entities}.")

    # Last known topic of conversation
    if ctx.last_focus:
        bits.append(f"Current focus: {ctx.last_focus}.")

    # Conversation topics/questions to revisit later
    if ctx.open_threads:
        top_open_threads = ", ".join(ctx.open_threads)
        bits.append(f"Open threads: {top_open_threads[:2]}.")

    # Topic of the current memory task (if there is one active)
    #if ctx.memory_topic:
    #    bits.append(f"Memory topic: {ctx.memory_topic}.")

    # If nothing in the context so far
    if not bits: return "Just met the user."

    # Hard cap to keep prompt short? (would be bits[:3])
    return " ".join(bits)


# ================================================================================
# Agent Configs & Client 
# ================================================================================
ROBOT_FAST = AgentConfig(
    name           = "robot_fast",
    model          = MODEL,
    temperature    = 0.7,
    max_tokens     = 128,
    window         =  2,
    response_model = RobotFastReply,  # Structured, but single field
)

ROBOT_SLOW = AgentConfig(
    name           = "robot_slow",
    model          = MODEL,
    temperature    = 0.5,
    max_tokens     = 256,
    window         =   2,
    response_model = RobotSlowUpdate,  # Deltas
)

USER_SIM = AgentConfig(
    name           = "user_sim",
    model          = MODEL,
    temperature    = 0.7,
    max_tokens     = 256,
    window         =   6,
    response_model = UserConversationResponse,
)

# Client Initialization (use in run_simulation)
def init_clients():
    client_raw = OpenAI(base_url=LLM_URL, api_key=LLM_KEY, timeout=20.0)
    client_structured = instructor.from_openai(
        OpenAI(base_url=LLM_URL, api_key=LLM_KEY, timeout=20.0),
        mode=instructor.Mode.JSON,
    )
    return client_raw, client_structured




# ================================================================================
# Helpers / Printers
# ================================================================================
def _hr(char: str = "-", width: int = 80) -> str:
    return char * width

def print_banner(title: str):
    print(f"\n{GREY}{BOLD}{_hr('=')}{RESET}")
    print(f"{GREY}{BOLD}{title}{RESET}")
    print(f"{GREY}{BOLD}{_hr('=')}{RESET}\n")

def print_turn_header(i: int):
    print(f"{GREY}{BOLD}================================ Turn {i} ================================ {RESET}\n")


# USER
def print_user_turn(duration: float, last_robot_message: str, user_message: str):
    print(f"{MAGENTA} --- USER RESPONSE ({duration:.2f}s) ------------------------------------ {RESET}")
    print(f"{YELLOW} Buddy:      {RESET}{last_robot_message}")
    print(f"{GREEN} Martha:     {RESET}{user_message}")
    print(f"{MAGENTA} {_hr('-')} {RESET}\n")

# ROBOT FAST
def print_fast_context(context_text: str):
    print(f"{RED} Context: {WHITE} {context_text}{RESET}")

def print_robot_fast(duration: float, message: str):
    print(f"{CYAN} --- ROBOT FAST TRACK ({duration:.2f}s) --------------------------------- {RESET}")
    print(f"{GREEN} Message:    {RESET}{message}")
    print(f"{CYAN} {_hr('-')} {RESET}\n")


# ROBOT SLOW
def print_robot_slow(duration: float, prev_state: str, slow_update: RobotSlowUpdate, ctx_before: ConversationContext, ctx_after: ConversationContext):
    print(f"{BLUE} --- ROBOT SLOW TRACK ({duration:.2f}s) [Controller Update] ------------- {RESET}")
    print(f"{WHITE_STD} Prev State: {RESET}{prev_state}")
    print(f"{WHITE_STD} New State:  {RESET}{slow_update.conversation_state}")
    print(f"{WHITE_STD} Reason:     {RESET}{slow_update.state_reason}")
    print(f"{WHITE_STD} Evidence:   {RESET}{slow_update.evidence}")
    print(f"{WHITE_STD} Deltas:     {RESET}{slow_update.context_delta}")
    print(f"{WHITE_STD} Plan:       {RESET}{slow_update.tentative_plan}")

    # Highlight if context changed
    if ctx_before.model_dump() != ctx_after.model_dump(): print(f"{LIME} Context:    {RESET}updated")
    else:                                                 print(f"{DIM}{LIME} Context:    {RESET}{DIM}no change{RESET}")

    print(f"{BLUE} {_hr('-')} {RESET}\n")

# JSON Context
def print_context_store(ctx: ConversationContext):
    print(f"{BLUE} --- New Context ---------------------------------")
    print(f"{WHITE}{ctx.model_dump_json(indent=2)}{RESET}")
    print(f"{BLUE} {_hr('-')} {RESET}\n")


# --------------------------------------------------------------------------------
# History Utilities
# --------------------------------------------------------------------------------
# User simulator "hears" Buddy as the 'user'
def _append_buddy_to_user_history(history_user: list, buddy_msg: str):
    history_user.append({"role": "user", "content": f"[Buddy]: {buddy_msg}"})

# User simulator "remembers" what it (Martha) said as the 'assistant'
def _append_martha_to_user_history(history_user: list, martha_msg: str):
    history_user.append({"role": "assistant", "content": martha_msg})


# --------------------------------------------------------------------------------
# State Gating
# --------------------------------------------------------------------------------
def gate_state(prev_state: ConversationState, proposed_state: ConversationState, ctx: ConversationContext) -> ConversationState:
    """ Only allow initiate_memory_activity after >=2 turns in interest. """
    if (proposed_state == "initiate_memory_activity"     ) and (prev_state != "explore_user_interests"  ) and (ctx.turns_in_interest < 2): return prev_state
    if (proposed_state == "discuss_memory_activity_topic") and (prev_state != "initiate_memory_activity") and (ctx.turns_in_interest < 2): return prev_state
    return proposed_state

def update_turns_in_interest(prev_state: ConversationState, new_state: ConversationState, ctx: ConversationContext):
    # Been in the same state for 2 in a row
    if   (prev_state == "explore_user_interests"       ) and (new_state == "explore_user_interests"       ): ctx.turns_in_interest += 1
    elif (prev_state == "initiate_memory_activity"     ) and (new_state == "initiate_memory_activity"     ): ctx.turns_in_interest += 1
    elif (prev_state == "discuss_memory_activity_topic") and (new_state == "discuss_memory_activity_topic"): ctx.turns_in_interest += 1

    # Reset count
    else:  ctx.turns_in_interest = 0


# ================================================================================
# Main Loop Function
# ================================================================================
def run_simulation(
    turns           : int   = 10,
    sleep_s         : float = 1.0,
    verbose_context : bool  = False,
):
    """
    Dual-track simulation:
    - USER_SIM: structured message-only
    - ROBOT_FAST: structured message-only (fast)
    - ROBOT_SLOW: structured controller update w/ deltas (sticky context/state)
    """
    print_banner("Starting Simulation: Split Model Architecture")

    # --------------------------------------------------------------------------------
    # 1) Initialize Clients & Histories
    # --------------------------------------------------------------------------------
    # Initialize clients
    client_raw, client_structured = init_clients()

    # Separate histories (user doesn't know about robots thought processes & vice versa)
    history_robot = [] # Buddy view: user/assistant turns (spoken only)
    history_user  = [] # Martha simulator view (Buddy as "user", Martha as "assistant")

    # --------------------------------------------------------------------------------
    # 2) Begin the Conversation (robot goes first)
    # --------------------------------------------------------------------------------
    # Initial State for the Robot
    start_message = "Good morning! My name is Buddy, it is nice to meet you. What is your name?"
    context       = init_context_store()
    current_state: ConversationState = "initiate_smalltalk"
    current_plan  = "Get the user's name and ask how they are doing."

    # First message
    print(f"{CYAN}BUDDY (Start):{RESET} {start_message}\n")

    # Update histories TODO: Need to make equivalent functions for robot
    history_robot.append({"role": "assistant", "content": start_message})
    _append_buddy_to_user_history(history_user, start_message)

    last_robot_message = start_message

    # ================================================================================
    # 3) Main Loop
    # ================================================================================
    for t in range(1, turns + 1):
        print_turn_header(t)

        # --------------------------------------------------------------------------------
        # a) USER Speaks (Grandma)
        # --------------------------------------------------------------------------------
        print(f"{MAGENTA} [USER] Thinking...{RESET}")
        t0 = time.time()

        # API call for response
        user_resp = run_agent(
            client_raw        = client_raw,
            client_structured = client_structured,
            agent             = USER_SIM,
            system_prompt     = USER_SYSTEM_PROMPT,
            history           = history_user,
        )
        t1 = time.time()

        martha_msg = user_resp.message.strip()
        print_user_turn(t1 - t0, last_robot_message, martha_msg)

        # Update histories
        history_robot.append({"role": "user", "content": martha_msg})
        _append_martha_to_user_history(history_user, martha_msg)

        # --------------------------------------------------------------------------------
        # b) ROBOT FAST (< 1.5s; robot's spoken reply)
        # --------------------------------------------------------------------------------
        # We inject the OLD state/context into the prompt to generate the reply quickly
        fast_context_text = render_context_for_fast(context)
        fast_sys_prompt   = get_robot_fast_prompt(state=current_state, context_text=fast_context_text, plan=current_plan)

        # Print context
        print(f"{CYAN} [ROBOT] Track 1: Generating reply...{RESET}")
        print_fast_context(fast_context_text)

        # API call for response
        t0 = time.time()
        fast_resp: RobotFastReply = run_agent(
            client_raw        = client_raw,
            client_structured = client_structured,
            agent             = ROBOT_FAST,
            system_prompt     = fast_sys_prompt,
            history           = history_robot,
        )
        t1 = time.time()

        # Print message
        buddy_msg = fast_resp.message.strip()
        print_robot_fast(t1 - t0, buddy_msg)

        # Update histories
        history_robot.append({"role": "assistant", "content": buddy_msg})
        _append_buddy_to_user_history(history_user, buddy_msg)

        last_robot_message = buddy_msg

        # --------------------------------------------------------------------------------
        # c) ROBOT SLOW (~5-8s) controller update (state + deltas + plan)
        # --------------------------------------------------------------------------------
        # Now we think about what just happened to prepare for the NEXT turn
        print(f"{BLUE} [ROBOT] Track 2: Updating state/context...{RESET}")

        # Handling context 
        ctx_before = ConversationContext.model_validate(context.model_dump())  # cheap snapshot copy
        slow_sys_prompt = get_robot_slow_prompt(current_state=current_state, context_json=context_to_json(context))

        # API call for response
        t0 = time.time()
        slow_update: RobotSlowUpdate = run_agent(
            client_raw        = client_raw,
            client_structured = client_structured,
            agent             = ROBOT_SLOW,
            system_prompt     = slow_sys_prompt,
            history           = history_robot,
        )
        t1 = time.time()

        # Apply deltas first
        context = apply_context_delta(context, slow_update.context_delta)

        # Gate state 
        proposed_state = slow_update.conversation_state
        gated_state = gate_state(current_state, proposed_state, context)

        # Update state counter
        update_turns_in_interest(current_state, gated_state, context)

        # Commit internal state/plan for next loop
        prev_state    = current_state
        current_state = gated_state
        current_plan  = slow_update.tentative_plan.strip()

        ctx_after = context

        print_robot_slow(t1 - t0, prev_state, slow_update, ctx_before, ctx_after)

        if verbose_context:  print_context_store(context)

        # Wait a little bit between turns
        if sleep_s and sleep_s > 0: time.sleep(sleep_s)


# ================================================================================
# Execution
# ================================================================================
print(f"{YELLOW}Attempting connection to: {LLM_URL}...{RESET}")
try: 
    run_simulation(
        turns           = 15,
        verbose_context = True,
    )

except Exception as e:
    print(f"\n{CYAN}--- CONNECTION ERROR ---{RESET}")
    print(f"{e}\n")
