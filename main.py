"""
main.py — Entry Point for the Semantic Field Detection System
=============================================================
How to run:
    python main.py

What it does:
    1. Shows a welcome header with project and author info.
    2. Lets you type any English sentence.
    3. Instantly shows the predicted Domain, Confidence, and XAI reasoning.
    4. Saves every result to results/session_log.txt automatically.
    5. Type 'quit' or 'exit' to stop.
"""

# ── Step 1: Import the libraries we need ────────────────────────────────────

import os                       # For creating folders
from datetime import datetime   # For timestamps in the log file

# Import the main detection function from our core module
from semantic_detector import explainable_semantic_detection


# ── Step 2: Set up constants (folder paths, header text) ────────────────────

# Where we store the log file
RESULTS_FOLDER = "results"
LOG_FILE_PATH = os.path.join(RESULTS_FOLDER, "session_log.txt")

# A simple line separator for clean output
LINE = "-" * 60

# The welcome banner users see when they start the program
WELCOME_HEADER = f"""
{LINE}
   SEMANTIC FIELD DETECTION SYSTEM
{LINE}
   Author  : Manyam Jagadeeswar Reddy (23BDS033)
   Course  : DS357 - Natural Language Processing (NLP)
   Dataset : Open English WordNet 2023
{LINE}
   Domains : MEDICAL | TECHNOLOGY | FINANCE | SPORTS | FOOD
{LINE}
   Type any English sentence and press Enter.
   Type 'quit' or 'exit' to stop.
{LINE}
"""


# ── Step 3: Helper function to create the results folder ────────────────────

def create_results_folder():
    """
    Create the 'results/' folder if it doesn't already exist.
    os.makedirs with exist_ok=True means it won't crash if the
    folder is already there.
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ── Step 4: Helper function to print the result nicely ──────────────────────

def print_result(result):
    """
    Take the result dictionary from explainable_semantic_detection()
    and print it in a clean, readable format.

    The result dictionary has these keys:
        - sentence    : the original input
        - domain      : the predicted domain (e.g. TECHNOLOGY)
        - confidence  : a percentage (e.g. 78.6)
        - status      : 'High confidence' or 'AMBIGUOUS (...)'
        - explanation  : list of strings showing which words drove the decision
    """
    print()
    print(f"   Domain     : [ {result['domain']} ]")
    print(f"   Confidence : {result['confidence']}%")
    print(f"   Status     : {result['status']}")

    # Only show the XAI trace if there are contributing keywords
    if result["explanation"]:
        trace = ", ".join(result["explanation"])
        print(f"   XAI Trace  : Decision driven by {trace}")

    print(LINE)


# ── Step 5: Helper function to save the result to a log file ────────────────

def save_to_log(result):
    """
    Append one analysis record to results/session_log.txt.
    Each record includes a timestamp so you can track when you ran it.
    """
    # Get the current date and time as a readable string
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open the file in 'append' mode ('a') so we add to it, never overwrite
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}]\n")
        log_file.write(f"  Input      : {result['sentence']}\n")
        log_file.write(f"  Domain     : {result['domain']}\n")
        log_file.write(f"  Confidence : {result['confidence']}%\n")
        log_file.write(f"  Status     : {result['status']}\n")

        if result["explanation"]:
            trace = ", ".join(result["explanation"])
            log_file.write(f"  XAI Trace  : {trace}\n")

        log_file.write(f"{LINE}\n")


# ── Step 6: The main loop — this is where everything runs ──────────────────

def main():
    """
    The main function that ties everything together:
      1. Create the results folder.
      2. Print the welcome header.
      3. Loop: ask for input → classify → display → log.
      4. Exit when the user types 'quit' or 'exit'.
    """

    # Make sure the results/ folder exists before we try to write to it
    create_results_folder()

    # Show the welcome banner
    print(WELCOME_HEADER)

    # Start the interactive loop
    while True:
        # Ask the user for a sentence
        try:
            sentence = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\n\n   Session ended.\n")
            break

        # Skip empty inputs (user just pressed Enter)
        if not sentence:
            continue

        # Check if the user wants to quit
        if sentence.lower() in ("quit", "exit"):
            print(f"\n   Session ended. Log saved to: {LOG_FILE_PATH}\n")
            break

        # ── Run the detection ───────────────────────────────────────────
        result = explainable_semantic_detection(sentence)

        # ── Show the result on screen ───────────────────────────────────
        print_result(result)

        # ── Save the result to the log file ─────────────────────────────
        save_to_log(result)


# ── Step 7: Standard Python entry point ─────────────────────────────────────
# This ensures main() only runs when you execute "python main.py" directly,
# not when someone imports this file as a module.

if __name__ == "__main__":
    main()
