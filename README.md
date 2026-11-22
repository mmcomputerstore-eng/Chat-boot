# Generative AI Practice Lab: LangChain & LangGraph

A hands-on practice project exploring modern **Generative AI**, **LangChain**, and **LangGraph**. This repository documents the concepts, patterns, and architectures I learned this week, ranging from basic prompt templates and vector embeddings to stateful graph workflows, tool-calling agents, and multi-agent coordination.

---

## 🚀 Overview of What I Learned

This practice repository covers key building blocks of LLM applications:

### 1. 🤖 Foundations & Chatbot Engineering
- **`chat_bot.py`**: Interactive terminal chatbot using Groq's high-speed inference (`openai/gpt-oss-120b`). Features real-time streaming, token usage calculation, cost estimation, prompt-length guards via `tiktoken`, and automatic conversation summarization every 5 turns.
- **`sentiment_analysis.py`**: Zero-shot sentiment classification using Hugging Face pipelines (`typeform/distilbert-base-uncased-mnli`) combined with few-shot LLM prompts.

### 2. 📑 Prompts & Structured Outputs
- **`prompt_template.py`**: Dynamic string prompt formatting using LangChain `PromptTemplate`.
- **`chat_prompt_template.py`**: Multi-role conversational prompt management using `ChatPromptTemplate` with `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`.
- **`structured_output.py`**: Forcing LLMs to return strict, typed JSON data validated through Pydantic models via `.with_structured_output()`.

### 3. 🧠 Embeddings & Semantic Search
- **`embedding.py`**: Generating text embeddings using `sentence-transformers/all-MiniLM-L6-v2`, computing manual cosine similarities with `util.cos_sim`, and running batch semantic search with `util.semantic_search`.

### 4. 📚 Retrieval-Augmented Generation (RAG)
- **`rag.py`**: Complete RAG pipeline:
  - Extracts and parses PDF documents (`symptoms.pdf`) via `pypdf`.
  - Splits text with `RecursiveCharacterTextSplitter`.
  - Generates normalized dense vector embeddings with HuggingFace.
  - Persists and queries vectors in a local **ChromaDB** store (`./chroma`).
  - Implements document deduplication using MD5 content hashing.
  - Answers user queries grounded strictly in retrieved medical context.

### 5. 🛠️ Tool-Calling Agents
- **`agent.py`**: Autonomous agent built with LangChain `create_agent` and custom Python functions decorated with `@tool` (`get_weather`). Handles model tool-calling loops, arguments parsing, and tool response injection.

### 6. 💾 Persistence & Session Checkpointing (LangGraph)
- **`checkpointers.py`**: Long-term state persistence using LangGraph's `PostgresSaver`.
  - Configures multi-tenant conversation threads (`thread_id`).
  - Supports live session switching (`session_1` vs `session_2`) with isolated memory per thread backed by a local PostgreSQL database.

### 7. ⚙️ Agent Middleware
- **`middleware.py`**: Attaches `SummarizationMiddleware` to an agent to automatically compress conversational context when token thresholds (e.g. 100 tokens) are exceeded, while keeping the most recent messages intact.

### 8. 🔄 LangGraph State Graphs & Cyclic Workflows
- **`graph.py`**: Stateful, multi-step LLM workflow built with LangGraph `StateGraph`:
  - Defines a typed state (`ContentState`) tracking drafts, ratings, feedback, and revision counters.
  - Chains nodes (`write_draft` ➔ `review_draft`).
  - Uses conditional edges (`should_continue`) to loop revisions until the content receives a score $\ge 8$ or reaches the maximum iteration limit.

### 9. 🤝 Multi-Agent Architecture
- **`multi_agent.py`**: Orchestration of specialized domain agents (`fruit_agent` vs `veggie_agent`), demonstrating routing and separation of responsibilities across distinct agents.

---

## 🛠️ Tech Stack & Tools

- **LLM Provider**: [Groq](https://groq.com) (LPU inference with `openai/gpt-oss-120b` and `openai/gpt-oss-20b`)
- **Frameworks**: LangChain, LangGraph
- **Vector Database**: ChromaDB (`langchain-chroma`)
- **Persistence Store**: PostgreSQL (`langgraph-checkpoint-postgres`, `psycopg`)
- **Embeddings & NLP**: `sentence-transformers`, `HuggingFaceEmbeddings`, `transformers`
- **Data Validation & Parsing**: Pydantic v2, `pypdf`, `tiktoken`

---

## 📁 Repository Structure

```text
├── agent.py                 # Tool-calling agent with @tool decorators
├── chat_bot.py              # Streaming terminal chatbot with cost & token tracking
├── chat_prompt_template.py  # ChatPromptTemplate practice
├── checkpointers.py         # LangGraph PostgreSQL session checkpointing
├── embedding.py             # Sentence embeddings and semantic search
├── graph.py                 # LangGraph state machine with feedback & revision loop
├── middleware.py            # Agent context compression via SummarizationMiddleware
├── multi_agent.py           # Multi-agent coordination and routing
├── prompt_template.py       # Basic LangChain PromptTemplate examples
├── rag.py                   # RAG pipeline with ChromaDB and PDF ingestion
├── sentiment_analysis.py    # Zero-shot classification & sentiment analysis
├── structured_output.py     # Pydantic structured output parsing
├── symptoms.pdf             # Sample knowledge document for RAG
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable template
└── .gitignore               # Configured to ignore .env, venv, and local chroma DB
```

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Chat boot"
```

### 2. Create and activate virtual environment
```bash
# Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the sample `.env.example` file:
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
DATABASE_URL=postgresql://user:password@localhost:5432/chathistory?sslmode=disable
```

---

## 🧪 Running Examples

You can run any practice module directly from your terminal:

```bash
# Run the interactive streaming chatbot:
python chat_bot.py

# Run the tool-calling agent:
python agent.py

# Run the LangGraph draft-and-critique state loop:
python graph.py

# Run the PostgreSQL multi-session checkpointer:
python checkpointers.py

# Run the RAG search over symptoms.pdf:
python rag.py

# Run semantic search with embeddings:
python embedding.py

# Run structured output extraction:
python structured_output.py
```

---

## 🔒 Security Note

Never commit your `.env` file containing private API keys or database credentials to GitHub. This repository's `.gitignore` is already preconfigured to protect `.env`, virtual environments, and local database directories.
