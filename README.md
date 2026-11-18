# Neon_bot - Groq-Powered AI Chatbot

A terminal-based AI chatbot built with Python and the OpenAI-compatible Groq API, featuring ultra-fast inference with the `openai/gpt-oss-120b` open-weights model, message memory, and real-time token/cost tracking.

---

## Features

- **Fast Inference**: Powered by Groq's high-speed LPU infrastructure.
- **Open-Weights Model**: Uses `openai/gpt-oss-120b`.
- **Interactive Chat Session**: Conversational loop directly in your terminal.
- **Usage & Cost Tracking**: Displays total prompt tokens, completion tokens, and estimated session cost.

---

## Prerequisites

- Python 3.10 or higher
- A Groq Cloud API key ([Get one free at console.groq.com](https://console.groq.com/keys))

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Chat boot"
```

### 2. Create and activate a virtual environment
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the sample environment file and add your Groq API key:
```bash
cp .env.example .env
```
Open `.env` in your editor and paste your API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

---

## Usage

Run the chatbot:
```bash
python chat_bot.py
```

- Type your question/prompt and press **Enter**.
- Type **`q`** and press **Enter** to quit the session.
- When you quit, the bot prints the total tokens consumed and estimated API cost.

---

## Project Structure

```text
├── chat_bot.py        # Main chatbot script
├── requirements.txt   # Python dependencies
├── .env.example       # Example environment configuration template
├── .gitignore         # Git ignore rules (protects your .env and virtual environment)
└── README.md          # Project documentation
```

---

## Security Note

Never commit your `.env` file containing your real `GROQ_API_KEY` to public repositories. The `.gitignore` file is already preconfigured to exclude `.env`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
