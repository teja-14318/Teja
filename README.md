import openai
import argparse
import time
import os
import sys
from datetime import datetime

# ===========================
# Configuration
# ===========================
openai.api_key = "your-api-key"  # Replace with your actual OpenAI API key

DEFAULT_MODEL = "gpt-3.5-turbo"
OUTPUT_FILE = "generated_texts.txt"
LOG_FILE = "generation_log.txt"

# ===========================
# Helper Functions
# ===========================

def log_message(message: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def save_output(prompt, response):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n=== Prompt ===\n{prompt}\n=== Response ===\n{response}\n{'='*40}\n")

def generate_text(prompt, model=DEFAULT_MODEL, max_tokens=300, temperature=0.7, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a creative text generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )
            result = response["choices"][0]["message"]["content"].strip()
            usage = response["usage"]
            log_message(f"Prompt processed (tokens: {usage['total_tokens']})")
            return result
        except openai.error.OpenAIError as e:
            log_message(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    return "Failed to generate text after multiple attempts."

def read_prompts_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

# ===========================
# Main CLI Logic
# ===========================

def main():
    parser = argparse.ArgumentParser(description="ChatGPT Text Generator (OpenAI API)")
    parser.add_argument("--prompt", type=str, help="Prompt to generate text from")
    parser.add_argument("--file", type=str, help="File with multiple prompts (one per line)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Creativity level (0-1)")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum tokens in output")

    args = parser.parse_args()

    prompts = []

    if args.prompt:
        prompts.append(args.prompt)
    elif args.file:
        prompts = read_prompts_from_file(args.file)
    else:
        prompt = input("Enter a prompt for ChatGPT: ")
        prompts.append(prompt)

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating text for prompt {i+1}/{len(prompts)}...")
        generated = generate_text(
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(f"\nPrompt: {prompt}\nResponse:\n{generated}\n")
        save_output(prompt, generated)

    print(f"\n✅ Done! Outputs saved to '{OUTPUT_FILE}' and logs to '{LOG_FILE}'")

# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    main()
