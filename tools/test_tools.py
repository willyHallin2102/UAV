"""
    tools / test_tools.py
    ---------------------
    Simple method for modularity from the test scripts.
"""
import argparse
import sys
import traceback

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List



@dataclass
class CommandSpec:
    """
    Declarative specification of a CLI command. Each command will defined following
    four attributes:
        - The command name, to call the command in terminal by.
        - A help description shown in argparse, help menu.
        - The callable handler to be executed when calling the command.
        - Optional, command-line argument definitions.
    
    The ``args`` field contains dictionaries compatible with 
    ``ArgumentParser.add_argument``
    
    Example
    -------
    >>> CommandSpec(
    ...     name="command-name",
    ...     help="Describe with this command",
    ...     handler=f,
    ...     args=[
    ...         {
    ...             "flags": ["--argument"],
    ...             "kwargs": {"type": int, "default": 10}
    ...         }
    ...     ]
    ... )
    -------
    """
    name    : str
    help    : str
    handler : Callable
    args    : List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """
            Validate command specification after dataclass initialization
        """
        if not self.name.isidentifier():
            raise ValueError(f"Command name ``{self.name}`` must be valid")
        
        if not callable(self.handler):
            raise TypeError(f"``{self.handler}`` must be callable to be called")



def builder(commands: List[CommandSpec]) -> argparse.Namespace:
    """
    Builds an argparse-based CLI parser from command specifications. The function
    dynamically creates sub-commands and attaches their corresponding argument
    definitions and handlers. Each command automatically receives:
        - Its own sup-parser
        - Configured command - line arguments
    -----
    Args:
    commands: List of ``CommandSpec`` representing a collection of command \
        specifications usd to construct the CLI interface.
    --------
    Returns:
    Fully configured parser containing all registered sub-commands and arguments.
    --------
    """
    parser = argparse.ArgumentParser(description="Logger, debug CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for command in commands:
        p = sub.add_parser(command.name, help=command.help)

        for arg in command.args:
            p.add_argument(*arg["flags"], **args["kwargs"])
        
        p.set_defaults(_handler=command.handler)
    
    return parser



def runner(f: Callable) -> Callable:
    """
    Wraps a CLI entrypoint with standardized exception handling. The wrapper ensures
        - Keyboard interrupts remain user-controlled
        - Runtime failures are printed in a consistent manner
        - Exception are re-raised for debugging visibility.
    -----
    Args:
    f: Callable target function that is getting wrapped
    --------
    Returns:
    Wrapped callable with centralized exception handling.
    --------
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        
        except KeyboardInterrupt:
            print("\nAborted by User")
            raise

        except Exception as e:
            print(f"\nTest failed:\t{e}")
            raise
    
    return wrapper
