"""
Helper functions for printing parts of the output.
--------------------------------------------------------------------------------
`src.utils.logging.utils`

"""
from .logging import RESET, BOLD, BRIGHT_GREY, BLACK


# Horizontal line
def hr(char: str = "-", width: int = 80) -> str:
    return char * width

# Title banner
def print_banner(title: str):
    print(
        f"{BLACK}{BOLD}{hr('=')}{RESET}\n"
        f"{BLACK}{BOLD}{title  }{RESET}\n"
        f"{BLACK}{BOLD}{hr('=')}{RESET}\n"
    )

# Turn separator
def print_turn_header(i: int):
    half_line = "=" * 35
    print(f"{BRIGHT_GREY}{BOLD}{half_line} Turn {i:03d} {half_line}{RESET}")

