"""
Neon_bot - Interactive Terminal AI Chatbot
Powered by Groq's ultra-fast inference API (OpenAI-compatible client).

Features:
- Real-time response streaming with a typewriter effect.
- Automatic context compression via conversation summarization every 5 turns.
- Message trimming utility for keeping context window manageable.
- Token counting and validation using tiktoken (cl100k_base).
- Session token usage and estimated API cost reporting.
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load environment variables from .env file (e.g., GROQ_API_KEY)
load_dotenv()


class ChatBot:
    """Manages chatbot state, API communication, message history, and token tracking."""

    def __init__(self):
        # Initialize the OpenAI client pointing to Groq's API endpoint
        self.llm = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        # Initialize conversation history with default persona / system prompt
        self.messages = [
            {
                "role": "system",
                "content": (
                    "you are a helpful assistant. your name is Neon_bot. "
                    "Introduce yourself in the first message. "
                    "give short and helpful answers"
                )
            }
        ]

        # Token usage counters for tracking session metrics & cost
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

        # Counter for total interaction turns
        self.iteration = 0

    def trim_messages(self, max_number: int = 4):
        """
        Trims conversation history to keep only the system prompt plus the most recent messages.
        
        Args:
            max_number: Maximum number of recent non-system messages to retain.
        """
        if len(self.messages) < max_number:
            return

        # Separate system prompts from conversational messages
        system_messages = [m for m in self.messages if m["role"] == "system"]
        other_messages = [m for m in self.messages if m["role"] != "system"]

        # Keep only the last 'max_number' messages
        trimmed_messages = other_messages[-max_number:]
        self.messages = system_messages + trimmed_messages

    def summarize_messages(self):
        """
        Periodically summarizes older conversation history to conserve context window.
        Runs every 5 conversation turns when at least 6 non-system messages exist.
        """
        other_messages = [m for m in self.messages if m["role"] != "system"]

        # Run only every 5 iterations and ensure enough history exists to compress
        if self.iteration == 0 or self.iteration % 5 != 0 or len(other_messages) < 6:
            return

        system_messages = [m for m in self.messages if m["role"] == "system"]

        # Take the oldest 6 messages (3 complete user-assistant conversation turns)
        early_messages = other_messages[:6]
        last_messages = other_messages[6:]

        # Request a concise summary from the LLM
        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize these conversation messages concisely, "
                        "keeping key facts, user preferences, and decisions."
                    )
                },
                *early_messages  # Unpack early messages into the message list
            ]
        )

        # Track tokens consumed by the summarization request
        if response.usage:
            self.total_tokens += response.usage.total_tokens
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens

        # Construct a system message containing the compressed summary
        summary_content = response.choices[0].message.content
        summary_message = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary_content}"
        }

        # Reconstruct message history: base system prompt + summary + recent messages
        self.messages = [*system_messages, summary_message, *last_messages]
        print("\n[System: Conversation history summarized successfully]")

    def bot(self, message: str) -> str:
        """
        Sends the user message to the LLM, streams the response to stdout,
        updates token counts, and appends the response to conversation history.

        Args:
            message: The user's input prompt.

        Returns:
            The complete response text from the assistant.
        """
        # Append user prompt to message history
        self.messages.append({
            "role": "user",
            "content": message
        })

        # Request a streaming response from the Groq API
        response = self.llm.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.messages,
            stream=True
        )

        response_text = ""

        # Stream chunks as they arrive for a real-time typewriter effect
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    response_text += delta
                    time.sleep(0.05)  # Slight delay to create smooth typing pace

            # Accumulate token usage when reported by stream chunks
            if chunk.usage:
                self.total_tokens += chunk.usage.total_tokens
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens

        # Record assistant reply in history
        self.messages.append({
            "role": "assistant",
            "content": response_text
        })

        # Trigger context summarization check
        self.summarize_messages()
        print()  # Print newline after stream ends
        return response_text

    def chat(self):
        """
        Runs the interactive terminal chat loop.
        Accepts user inputs, enforces input token limits, and quits on 'q'.
        """
        # Initialize BPE tokenizer for prompt length validation
        encoding = tiktoken.get_encoding("cl100k_base")

        while True:
            self.iteration += 1
            user_input = input('(Type "q" to quit) Ask Something : ').strip().lower()

            # Exit condition
            if user_input == "q":
                print("Good luck")
                break

            # Calculate prompt token length
            token_count = len(encoding.encode(user_input))

            # Guard against overly long user inputs (>100 tokens)
            if token_count > 100:
                print("Your prompt exceeded the limit. Try a shorter one, please")
                continue

            # Process the message
            self.bot(user_input)

    def cost(self):
        """
        Calculates and prints estimated API cost based on token counts.
        Rate formula: $0.05 / 1M prompt tokens + $0.08 / 1M completion tokens.
        """
        estimated_cost = (self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000
        print(f"Cost : ${estimated_cost:.6f}")


def main():
    """Application entry point: instantiates chatbot and starts interactive session."""
    chat = ChatBot()
    chat.chat()

    # Print summary metrics upon exit
    print(f"Consumed tokens: {chat.total_tokens}")
    chat.cost()


if __name__ == "__main__":
    main()
