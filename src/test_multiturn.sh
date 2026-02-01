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
cp ../instructions.json .

# ================================================================================
# 3. Create Python Script
# ================================================================================
log_step "3" "Generating Python client code (main.py)"
cat << 'EOF' > main.py
import os
import time
import json
import instructor

from pydantic import BaseModel, Field
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
llm_key = os.getenv("LLM_KEY", "SAMPLE_TOKEN")

# MODEL SELECTION
# Options: phi3-buddy | phi3.5-mini | qwen2.5-3b | qwen2.5-3b-speculative | qwen2.5-0.5b
MODEL = "phi3.5-mini"

# --------------------------------------------------------------------------------
# Scenario Prompting (NEW)
# --------------------------------------------------------------------------------
# read the scenario instructions from the mounted file path.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTRUCTIONS_JSON_PATH = os.getenv(
    "INSTRUCTIONS_JSON",
    os.path.join(SCRIPT_DIR, "instructions.json")
)

# Starting scenario
START_SCENARIO = os.getenv("START_SCENARIO", "start_conversation")

def load_scenarios(path: str) -> list[dict]:
    """Load scenario list from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("instructions.json must be a JSON list of scenario objects.")
    return data

def format_available_scenarios(scenarios: list[dict]) -> str:
    """
    Turn:
      [{"name": "...", "short_description": "..."}]
    into:
      - name: short_description
    """
    lines = []
    for s in scenarios:
        name = s.get("name", "").strip()
        desc = (s.get("short_description") or "").strip()
        if name:
            lines.append(f"- {name}: {desc}")
    # Ensure start scenario is always present
    if not any((s.get("name") or "").strip() == START_SCENARIO for s in scenarios):
        lines.insert(0, f"- {START_SCENARIO}: starting state of the conversation")
    return "\n".join(lines)

def get_instruction_text(scenarios: list[dict], current_scenario: str) -> str:
    """Find the instruction blob for the current scenario."""
    for s in scenarios:
        if (s.get("name") or "").strip() == current_scenario:
            return (s.get("instruction") or "").strip()
    # Fallback if scenario not found
    return "No instructions found for this scenario. Stay in the current scenario and respond briefly."

def build_system_prompt(*, available_scenarios_text: str, current_scenario: str, instructions_text: str) -> str:
    """
    NEW: A stricter, simpler prompt focused on:
      - generating assistant_response
      - predicting next_scenario
    """
    return f"""
You are a scenario-based conversational assistant.

The conversation is structured into SCENARIOS (stages). You will be given:
- AVAILABLE_SCENARIOS (names + descriptions)
- CURRENT_SCENARIO
- INSTRUCTIONS for CURRENT_SCENARIO

AVAILABLE_SCENARIOS:
{available_scenarios_text}

CURRENT_SCENARIO: "{current_scenario}"

INSTRUCTIONS FOR CURRENT_SCENARIO:
----------------
{instructions_text}
----------------

Your job each turn:
1) Read the user's most recent message AND the chat history.
2) Follow the CURRENT_SCENARIO instructions to write the best next reply.
3) Decide NEXT_SCENARIO:
   - If the goals of CURRENT_SCENARIO are NOT met, keep next_scenario == CURRENT_SCENARIO.
   - If the goals ARE met, choose the best next scenario from AVAILABLE_SCENARIOS.

STRICT OUTPUT CONTRACT:
- Output exactly ONE JSON object and nothing else.
- No markdown, no code fences, no extra text.
- Keys must be exactly: "assistant_response", "next_scenario"
- "next_scenario" must be either CURRENT_SCENARIO or one of AVAILABLE_SCENARIOS.
""".strip()

# --------------------------------------------------------------------------------
# Pydantic Output Model (CHANGED)
# --------------------------------------------------------------------------------
class ScenarioResponse(BaseModel):
    assistant_response: str = Field(..., description="Short natural language reply to the user's last message.")
    next_scenario: str = Field(..., description="Scenario name to use for the next turn.")

# --------------------------------------------------------------------------------
# Helpers (NEW)
# --------------------------------------------------------------------------------
def validate_next_scenario(next_scenario: str, allowed: set[str], current: str) -> tuple[bool, str]:
    """Validate next_scenario against allowed set; return (ok, error_message)."""
    if next_scenario == current:
        return True, ""
    if next_scenario in allowed:
        return True, ""
    return False, f'next_scenario="{next_scenario}" is not in allowed scenarios (or current scenario).'

def print_history(messages: list[dict]):
    """Pretty print the message history we're sending (excluding system prompt)."""
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").replace("\n", " ")
        if len(content) > 140:
            content = content[:140] + "..."
        print(f"  - {role}: {content}")

# --------------------------------------------------------------------------------
# Get a response from the LLM (CHANGED: now accepts full messages list)
# --------------------------------------------------------------------------------
def get_response(client, *, system_prompt: str, messages: list[dict], allowed_scenarios: set[str], current_scenario: str, label: str):
    print(f"{CYAN}Sending request...{RESET} {YELLOW}({label}){RESET}")

    print(f"{YELLOW}History being sent (excluding system):{RESET}")
    print_history(messages)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            response_model=ScenarioResponse,
            temperature=0.3,
            max_tokens=512,
        )
        t1 = time.time()
        duration = t1 - t0

        ok, err = validate_next_scenario(response.next_scenario, allowed_scenarios, current_scenario)

        print(f"{CYAN}--- MODEL RESPONSE ({duration:.2f}s) ---{RESET}")
        print(f"{GREEN}assistant_response:{RESET} {response.assistant_response}")
        print(f"{GREEN}next_scenario:     {RESET} {response.next_scenario}")
        if not ok:
            print(f"{RED}VALIDATION ERROR:  {RESET} {err}")
        print(f"{CYAN}-------------------------------{RESET}\n")

    except Exception as e:
        print(f"{CYAN}--- ERROR ---{RESET}")
        print(f"{e}\n")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
print(f"{YELLOW}Attempting connection to: {llm_url}...{RESET}")
print(f"{YELLOW}Model endpoint: {MODEL}{RESET}")
print(f"{YELLOW}Instructions file: {INSTRUCTIONS_JSON_PATH}{RESET}")

try:
    scenarios = load_scenarios(INSTRUCTIONS_JSON_PATH)
    allowed_scenarios = { (s.get("name") or "").strip() for s in scenarios if (s.get("name") or "").strip() }
    if START_SCENARIO not in allowed_scenarios:
        allowed_scenarios.add(START_SCENARIO)

    available_scenarios_text = format_available_scenarios(scenarios)

    # We'll run tests starting from the start scenario by default.
    # (You can override this via env if you want.)
    current_scenario = os.getenv("CURRENT_SCENARIO", START_SCENARIO)

    base_instruction_text = get_instruction_text(scenarios, current_scenario)

    # NEW: Inflate instructions to test large system prompts.
    instructions_text = (base_instruction_text + "\n\n")

    system_prompt = build_system_prompt(
        available_scenarios_text=available_scenarios_text,
        current_scenario=current_scenario,
        instructions_text=instructions_text,
    )

except Exception as e:
    print(f"{RED}Failed to load/build scenario prompt:{RESET} {e}")
    raise

try:
    client = instructor.from_openai(
        OpenAI(
            base_url=llm_url,
            api_key=llm_key,
            timeout=20.0,
        ),
        mode=instructor.Mode.JSON
    )

    # -------------------------------------------------------------------------
    # TEST 1: 0-message history (system + 1 user)
    # -------------------------------------------------------------------------
    messages_0 = [
        {"role": "user", "content": "Hello!"},
    ]
    get_response(
        client,
        system_prompt=system_prompt,
        messages=messages_0,
        allowed_scenarios=allowed_scenarios,
        current_scenario=current_scenario,
        label="TEST 1: 0-history",
    )

    # -------------------------------------------------------------------------
    # TEST 2: 5-turn history (10 messages user/assistant) + final user question
    # -------------------------------------------------------------------------
    # The goal is to ensure the LLM handles multiple user/assistant turns correctly.
    messages_5_turns = [
        {"role": "user", "content": "Hi QT!"},
        {"role": "assistant", "content": "Hi there! I’m QT Robot, a friendly social robot. What should I call you?"},
        {"role": "user", "content": "You can call me Ana."},
        {"role": "assistant", "content": "Nice to meet you, Ana! What’s something small that made you smile today?"},
        {"role": "user", "content": "My granddaughter visited, it was sweet."},
        {"role": "assistant", "content": "That sounds really special. What did you two do together?"},
        {"role": "user", "content": "We looked at old photos and laughed."},
        {"role": "assistant", "content": "Old photos can bring back warm memories. What was one photo that stood out to you?"},
        {"role": "user", "content": "A picture of our first house from the 70s."},
        {"role": "assistant", "content": "That must have felt nostalgic. What do you remember most about living there?"},
        # The *current* user message (the one we want the model to answer now):
        {"role": "user", "content": "It felt simpler back then. Why do you think that is?"},
    ]

    current_scenario = "explore_user_interests"

    system_prompt_2 = build_system_prompt(
        available_scenarios_text=available_scenarios_text,
        current_scenario=current_scenario,
        instructions_text=instructions_text,
    )
    get_response(
        client,
        system_prompt=system_prompt_2,
        messages=messages_5_turns,
        allowed_scenarios=allowed_scenarios,
        current_scenario=current_scenario,
        label="TEST 2: 5-turn history",
    )

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
COPY instructions.json .
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
