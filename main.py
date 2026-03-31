"""
Semantic Field Detection — Interactive CLI
===========================================
Course  : DS357 – Natural Language Processing (NLP)
Author  : Manyam Jagadeeswar Reddy (23BDS033)
Institution : IIIT Dharwad

Usage:
    python main.py

Description:
    Interactive terminal interface for the Semantic Field Detection system.
    Allows the user to type English sentences and receive real-time domain
    classification with confidence scores and explainable reasoning traces.
    Every analysis is logged to ``results/session_log.txt``.
"""

import os
import sys
from datetime import datetime

from semantic_detector import explainable_semantic_detection

# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "session_log.txt")

HEADER = r"""
========================================================================
    SEMANTIC FIELD DETECTION SYSTEM
========================================================================
    Author  : Manyam Jagadeeswar Reddy (23BDS033)
    Course  : DS357 - Natural Language Processing (NLP)
    Dataset : Open English WordNet 2023  (161,338 words | 120,135 synsets)
========================================================================
    Domains : MEDICAL | TECHNOLOGY | FINANCE | SPORTS | FOOD
------------------------------------------------------------------------
    Type a sentence and press Enter to classify.
    Type 'quit' or 'exit' to end the session.
========================================================================
"""

SEPARATOR = "-" * 72


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _ensure_results_dir() -> None:
    """Create the results/ directory if it does not already exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _log_result(result: dict) -> None:
    """
    Append a formatted analysis record to the session log file.

    Parameters
    ----------
    result : dict
        Output dictionary from ``explainable_semantic_detection()``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"  Input      : {result['sentence']}\n")
        f.write(f"  Domain     : {result['domain']}\n")
        f.write(f"  Confidence : {result['confidence']}%\n")
        f.write(f"  Status     : {result['status']}\n")
        if result["explanation"]:
            f.write(
                f"  XAI Trace  : {', '.join(result['explanation'])}\n"
            )
        f.write(f"{SEPARATOR}\n")


def _display_result(result: dict) -> None:
    """
    Pretty-print the classification result to the terminal.

    Parameters
    ----------
    result : dict
        Output dictionary from ``explainable_semantic_detection()``.
    """
    print(f"\n  Domain     : [ {result['domain']} ]")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Status     : {result['status']}")
    if result["explanation"]:
        print(
            f"  XAI Trace  : Decision driven by {', '.join(result['explanation'])}"
        )
    print(SEPARATOR)


# =============================================================================
# MAIN LOOP
# =============================================================================

def main() -> None:
    """Run the interactive classification loop."""

    _ensure_results_dir()

    print(HEADER)

    while True:
        try:
            sentence = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Graceful exit on Ctrl-C / Ctrl-D
            print("\n\n  Session terminated.\n")
            break

        if not sentence:
            continue

        if sentence.lower() in {"quit", "exit"}:
            print("\n  Session ended. Results saved to:", LOG_FILE, "\n")
            break

        result = explainable_semantic_detection(sentence)

        _display_result(result)
        _log_result(result)


if __name__ == "__main__":
    main()
