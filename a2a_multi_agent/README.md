# Multi-Agent A2A Demo

This is a conceptual demo of a multi-agent system built using Google’s Agent-to-Agent (A2A) protocol, LangChain, and Gemini models.

The setup demonstrates how a Streamlit-based client can send a user query to a central **Orchestrator Agent**, which then delegates the task to specialized A2A-compatible agents. The orchestrator uses simple routing logic to determine whether to invoke a **Greeting Agent** or a **Time Agent**, then collects the response and returns it to the client.

This repo serves as a starting point to experiment with inter-agent collaboration, skill-based task routing, and streaming responses in an A2A-driven architecture.

## Agents Involved

This demo features three independent agents, each implemented as an A2A-compatible service with LangChain and Gemini under the hood:

- 🧠 **Orchestrator Agent**  
  Acts as the central decision-maker. It receives the user's query from the client and routes it to the appropriate downstream agent. It does not generate content directly, but delegates the task and aggregates the response.

- 💬 **Greeting Agent**  
  Replies to the user with a short, warm greeting followed by an inspirational quote. It uses Gemini 2.0 Flash and is designed to simulate a pleasant AI persona. All logic is encapsulated in the `GreetingQuoteAgent` class.

- 🕒 **Time Agent**  
  Returns the current time based on the user's mentioned country or timezone context. This agent also runs on Gemini and handles simple time localization tasks.

Each agent runs as a separate HTTP service and exposes its capabilities via A2A’s standardized task interface.

## Architecture Overview

This demo follows a simple multi-agent architecture:

1. A **Streamlit frontend** acts as the user interface.
2. The **Orchestrator Agent** receives the input query and determines which helper agent to delegate to.
3. Based on intent, the orchestrator makes an A2A HTTP call to either the **Greeting Agent** or the **Time Agent**.
4. The selected agent responds with a structured result, which is returned to the client.

All communication between agents follows **Google’s A2A protocol** using **JSON-RPC 2.0 over HTTP**, served via lightweight **Starlette applications** running on **Uvicorn**.  
This enables easy streaming, structured task updates, and standard artifact handling between services.

---

### 📁 Folder Structure

```bash
a2a_multi_agent/
│
├── agents/
│   ├── greet_agent/
│   │   ├── __main__.py             # Entrypoint to run the Greeting Agent
│   │   ├── agent_executor.py       # A2A bridge: handles task lifecycle
│   │   └── quote_agent.py          # LangChain Gemini-based quote generation
│   │
│   ├── time_agent/
│   │   ├── __init__.py
│   │   ├── __main__.py             # Entrypoint for Time Agent
│   │   ├── agent_executor.py       # A2A executor for time responses
│   │   └── agent.py                # LangChain Gemini time logic
│   │
│   └── orchestrator/
│       ├── __main__.py             # Entrypoint for Orchestrator Agent
│       ├── executor.py             # Central A2A logic for routing
│       └── routing_agent.py        # Routing logic
│
├── client/
│   └── streamlit_app.py            # Minimal Streamlit frontend (sends user input)
```

## Code Structure

Each agent is self-contained with a clear separation of concerns between orchestration, execution logic, and the actual agent behavior. Here's a breakdown of the key components:

### 🔹 `agents/greet_agent/`
- **`__main__.py`** Launches the Greeting Agent via Uvicorn.
- **`agent_executor.py`** Bridges A2A task lifecycle with `GreetingQuoteAgent`. Handles streaming and final task artifacts.
- **`quote_agent.py`** Defines the Gemini-powered logic to generate a greeting and an inspirational quote.

### 🔹 `agents/time_agent/`
- **`__main__.py`** Entry script to start the Time Agent service.
- **`agent_executor.py`** Handles A2A task execution for time requests.
- **`agent.py`** Contains LangChain + Gemini logic to respond with current time based on country or timezone context.

### 🔹 `agents/orchestrator/`
- **`__main__.py`** Launches the Orchestrator Agent.
- **`executor.py`** Central handler for incoming A2A tasks. Uses `routing_agent.py` to determine the appropriate target agent.
- **`routing_agent.py`** Contains intent recognition logic and routing rules (e.g., route to greeting vs time).

### 🖥️ `client/`
- **`streamlit_app.py`** Minimal frontend to send user queries to the Orchestrator Agent. Receives and displays the response from downstream agents.

Each agent uses A2A's `AgentCard`, task streaming, and artifact APIs to expose a clean and interoperable interface for collaboration.

## How to Run the Demo

Follow the steps below to set up and run all components locally.

### 🧱 1. Clone the Repository

```bash
git clone https://github.com/aman-panjwani/a2a_demos.git
cd a2a_demos/a2a_multi_agent
```

### 🐍 2. Set Up the Python Environment

This project uses [**uv**](https://github.com/astral-sh/uv) for environment and package management:

```bash
uv init --python python3.13
uv venv
```

Activate the virtual environment:

- **Windows**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux**
  ```bash
  source .venv/bin/activate
  ```

Then install dependencies:

```bash
uv pip install -r requirements.txt
```

### 🔐 3. Configure Environment Variables

Create a `.env` file in the root of `a2a_multi_agent` and copy values from `.env.example`. At minimum, set:

```env
GOOGLE_API_KEY="your_google_api_key_here"
A2A_PEERS=http://localhost:10000,http://localhost:10001
```

### 🤖 4. Start Each Agent (In Separate Terminals)

Each agent should be run in its own terminal:

#### 💬 Greeting Agent

```bash
uv run python -m agents.greet_agent
```

#### 🕒 Time Agent

```bash
uv run python -m agents.time_agent
```
#### 🧠 Orchestrator Agent

```bash
uv run python -m agents.orchestrator
```


When an agent starts, you’ll see logs like:

```
INFO:     Uvicorn running on http://localhost:10001 (Press CTRL+C to quit)
```

Each agent runs on a dedicated port, as defined in the `main()` function.

### 🖥️ 5. Run the Streamlit Client

Once all agents are running, launch the frontend:

```bash
streamlit run client/streamlit_app.py
```

This will open a browser window where you can send prompts and observe how the orchestrator routes the tasks to the appropriate agents.

---

### 🎬 Demo Video

Watch the demo in action:

[▶️ Click to watch the demo](assets/multi_agent_demo.mp4)

## Key Features Demonstrated

This multi-agent demo showcases several core concepts enabled by the A2A protocol and the surrounding ecosystem:

- 🔁 **Multi-agent orchestration**  
  The Orchestrator Agent receives a task and intelligently routes it to the appropriate helper agent (Greeting or Time) based on intent.

- 🌐 **Agent-to-agent communication via A2A**  
  Agents interact using Google's A2A protocol with JSON-RPC 2.0 over HTTP, enabling structured, modular, and scalable agent interaction.

- 📡 **Streaming and task status updates**  
  Agents stream partial responses and status updates during task execution using the A2A runtime's event queue system.

- 🧠 **LangChain + Gemini integration**  
  Each agent wraps a LangChain-compatible Gemini model (via `ChatGoogleGenerativeAI`) to handle natural language tasks.

- ⚙️ **Self-contained Starlette + Uvicorn services**  
  Each agent runs as a lightweight HTTP service using Starlette and Uvicorn, exposing its `AgentCard`, skills, and capabilities independently.

- 🧪 **Experiment-ready design**  
  The entire setup is modular and minimal, designed to test patterns like delegation, streaming, and centralized routing in an A2A environment.


## Ideas & Use Cases

While this is a conceptual demo, the multi-agent architecture showcased here maps well to real-world production scenarios. Below are practical ideas and enterprise-grade use cases that can extend from this foundation:

### Realistic Production Use Cases

- **AI Agent Hubs for Enterprise Teams**  
  Route queries across specialized agents (e.g., HR, Finance, IT) that access different data systems securely and independently.

- **AI-Powered Customer Support Systems**  
  Deploy domain-specific agents (billing, tech support, onboarding) behind a single conversational interface. The orchestrator agent routes queries based on customer intent and urgency.

- **Internal Knowledge Assistants**  
  Connect multiple agents to internal knowledge bases (e.g., Confluence, SharePoint, Notion) and route based on department or topic enabling a unified AI knowledge layer across the org.

- **AI Content Assembly Pipelines**  
  Use different agents for outlining, drafting, fact-checking, and formatting orchestrated dynamically based on document type or workflow stage (great for marketing or RFP automation).

### Extend This Demo To:
- Add new skills to existing agents (e.g., sentiment detection in greeting agent).
- Introduce authentication and scoped access control per agent.
- Deploy agents as containerized microservices with internal service mesh (e.g., Istio).
- Implement a discovery registry where agents dynamically advertise their skills.

This demo lays the groundwork for these architectures by modeling how lightweight, task-focused agents can collaborate something that scales well in distributed environments.

## Final Thoughts

This demo is a foundation not a framework.

The goal is to explore how simple, focused agents can collaborate effectively using A2A. Whether you're experimenting with intelligent assistants, building internal tools, or architecting distributed AI systems, this setup gives you a flexible starting point.

Feel free to fork, remix, or extend. And if you find something useful or build something on top I’d love to hear about it.

Stay curious, and keep building 🛠️

---

If this project helped or inspired you, consider giving it a ⭐ on GitHub it helps others discover it!
