import os
import sys
import platform
import subprocess
from pathlib import Path
from chatbot.chatbot import Chatbot

# Evaluates the parent of the current working directory, then looks into the 'pkl' folder
model_path = Path(__file__).resolve().parent /"pkl" / "custom_transformer_model.pth"


# Check if model exists
if model_path.exists():
    print("Model found.")
else:
    print("Model not found. Running Main.py...")

    # Detect OS
    system_type = platform.system()

    # Path to Main.py
    main_script = (
        Path(__file__).resolve().parent
        / "model/Main.py"
    )

    # Build command
    if system_type == "Windows":
        command = ["python", str(main_script)]
    else:
        command = ["python3", str(main_script)]

    print("Running:", " ".join(command))

    # Execute script
    subprocess.run(command)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and chat with your custom transformer model')
    parser.add_argument('--model', type=str, default=model_path,
                       help='Path to your trained model file')
    parser.add_argument('--simple', action='store_true',
                       help='Simple chat mode (no typing effect)')
    
    args = parser.parse_args()
    
    # Create chatbot
    bot = Chatbot(args.model)
    
    if args.simple:
        # Simple mode without typing effect
        print("\nSimple chat mode (type 'quit' to exit)\n")
        while True:
            user = input("You: ")
            if user.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            if user:
                print(f"Assistant: {bot.generate_response(user)}\n")
    else:
        # Full chat mode
        bot.chat()