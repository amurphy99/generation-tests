"""
Helper functions for printing parts of the output.
--------------------------------------------------------------------------------
`src.utils.logging.utils`

"""
from .logging import RESET, BOLD, BRIGHT_GREY



def _hr(char: str = "-", width: int = 80) -> str:
    return char * width

def print_banner(title: str):
    print(f"\n{BRIGHT_GREY}{BOLD}{_hr('=')}{RESET}")
    print(f"{BRIGHT_GREY}{BOLD}{title}{RESET}")
    print(f"{BRIGHT_GREY}{BOLD}{_hr('=')}{RESET}\n")

def print_turn_header(i: int):
    print(f"{BRIGHT_GREY}{BOLD}================================ Turn {i} ================================ {RESET}\n")
