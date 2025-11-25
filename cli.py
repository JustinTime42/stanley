#!/usr/bin/env python
"""
Agent Swarm CLI entry point.

Usage:
    python cli.py                    # Start interactive REPL
    python cli.py "query"            # Start with initial prompt
    python cli.py -p "query"         # One-shot mode
    python cli.py -c                 # Continue last session
    python cli.py -r <session-id>    # Resume specific session
"""

from src.cli.app import main

if __name__ == "__main__":
    main()
