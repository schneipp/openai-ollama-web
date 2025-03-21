# 🤖 AI Agents Framework 🚀

## Overview

This project demonstrates how to build and deploy a multi-agent AI system using OpenAI-compatible models. The framework enables specialized AI agents to collaborate, with each agent handling specific tasks like web searching, location detection, news gathering, and more.

## ✨ Features

- 🧠 **Triage Agent**: Orchestrates requests between specialized agents
- 🔍 **Web Search Agent**: Fetches information from the web using DuckDuckGo
- 📰 **News Search Agent**: Retrieves the latest news on specified topics
- 📍 **Location Assistant**: Provides (fake) geographic location data
- 📅 **Date and Day Agent**: Delivers current date and day information
- 🔄 **Format and Translate Agent**: Translates content to German and formats with Markdown

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- An OpenAI-compatible API endpoint (local or remote)

### Environment Variables

Set the following environment variables or modify them directly in the code:

```bash
EXAMPLE_BASE_URL="http://localhost:11434/v1"  # Your API endpoint
EXAMPLE_API_KEY="your_api_key"                # Your API key
EXAMPLE_MODEL_NAME="model_name"               # Model to use
```

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install openai-agents duckduckgo_search
```

## 🚀 Usage

Run the script and enter your query when prompted:

```bash
python main.py
```

The system will:

1. Process your input through the triage agent
2. Dispatch specialized agents as needed
3. Return a formatted response that may include web search results, location data, or current date information

## 🧩 Architecture

This project uses a composable agent architecture:

- **Base Tools**:
  - `get_weather`: Location information
  - `websearch`: Web search functionality
  - `newssearch`: News search functionality
  - `get_date_and_day`: Date and time information

- **Specialized Agents**:
  Each agent has specific instructions and access to relevant tools

- **Triage Agent**:
  Coordinates between specialized agents to handle complex queries

## 🔧 Customization

To add new functionality:

1. Create a new function tool using the `@function_tool` decorator
2. Create a new agent with relevant instructions and tools
3. Add the new agent to the triage agent's handoffs list

## 📝 Example

Input:

```
What's happening in tech today?
```

Output:

```markdown
# 🖥️ Tech-Neuigkeiten des Tages 🚀

Heute ist **Freitag, 21.03.2025**, und hier sind die aktuellsten Entwicklungen in der Tech-Welt:

- 📱 **Apple plant neue VR-Headset-Generation** - Berichten zufolge mit verbesserter Akkulaufzeit und leichterem Design
- 🤖 **Fortschritte in der KI-Forschung** - Neue Modelle zeigen verbesserte Fähigkeiten bei komplexem Reasoning
- 🔒 **Wichtige Sicherheitsupdates** - Mehrere Tech-Unternehmen veröffentlichen kritische Patches

*Alle Informationen stammen aus aktuellen Online-Quellen und wurden für Sie zusammengefasst.*
```

## 📄 License

[MIT License](LICENSE)
