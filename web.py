import asyncio
import os
from duckduckgo_search import DDGS
import datetime
import gradio as gr
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import json

from openai import AsyncOpenAI

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)

BASE_URL = os.getenv("EXAMPLE_BASE_URL") or ""
API_KEY = os.getenv("EXAMPLE_API_KEY") or ""
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME") or ""

BASE_URL = "http://localhost:11434/v1"
API_KEY = "fake"
# MODEL_NAME = "hf.co/mradermacher/DeepSeek-R1-Distill-Llama-3B-tools-GGUF:latest"
# MODEL_NAME = "MFDoom/deepseek-r1-tool-calling"
# MODEL_NAME = "llama3.1:latest"
MODEL_NAME = "jacob-ebey/phi4-tools:latest"
# MODEL_NAME = "mistral"

if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set EXAMPLE_BASE_URL, EXAMPLE_API_KEY, EXAMPLE_MODEL_NAME via env var or code."
    )

"""This example uses a custom provider for a specific agent. Steps:
1. Create a custom OpenAI client.
2. Create a `Model` that uses the custom client.
3. Set the `model` on the Agent.

Note that in this example, we disable tracing under the assumption that you don't have an API key
from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
or call set_tracing_export_api_key() to set a tracing specific key.
"""
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)

# An alternate approach that would also work:
# PROVIDER = OpenAIProvider(openai_client=client)
# agent = Agent(..., model="some-custom-model")
# Runner.run(agent, ..., run_config=RunConfig(model_provider=PROVIDER))


@function_tool
def get_weather(city: str):
    print(f"[debug] returning the location")
    return f"the location is Schaffhausen, Switzerland, Europe"


@function_tool
def websearch(query: str):
    print(f"[debug] doing web search on duckduckgo_search {query}")
    ddgs_results = DDGS().text(
        query, region="wt-wt", safesearch="off", timelimit="n", max_results=4
    )
    print(f"[debug] search is done")
    results = ""
    for result in ddgs_results:
        results += result["title"] + " - " + result["body"] + "\n"
    print(f"[debug] doing web search on duckduckgo_search {query}")
    return results


@function_tool
def newssearch(query: str):
    print(f"[debug] doing news search on duckduckgo_search {query}")
    ddgs_results = DDGS().news(
        query, region="wt-wt", safesearch="off", timelimit="n", max_results=10
    )
    print(f"[debug] news search is done")
    results = ""
    for result in ddgs_results:
        results += result["title"] + " - " + result["body"] + "\n"
    return results


@function_tool
def get_date_and_day(query: str):
    print(f"[debug] populate data {query}")
    date = datetime.datetime.now()
    return (
        "the current date is: "
        + date.strftime("%d.%m.%Y")
        + ", heute ist "
        + date.strftime("%A")
        + " prefered language is german"
    )


location_agent = Agent(
    name="Location Assistant",
    instructions="You populate the users request with the exact geographic location",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_weather],
)

web_agent = Agent(
    name="Websearch Agent",
    instructions="You search the web for topics and use the result to answer the question asked.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[websearch],
)

news_agent = Agent(
    name="News Search Agent",
    instructions="You search the web for the latest news and use the result to answer the question asked.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[newssearch],
)

date_and_day_agent = Agent(
    name="Get Date And Day",
    instructions="You return the current date and day to the users request if needed",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[get_date_and_day],
)

format_agent = Agent(
    name="Format and Translate Agent",
    instructions="you translate the request to german, add eomjois and format it as markdown",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "first you get the current date and day"
        "You triage the user's request and provide a response."
        "translate the request to german, add emojis and format it as valid markdown"
    ),
    handoffs=[date_and_day_agent, location_agent, web_agent, news_agent, format_agent],
    tools=[get_date_and_day, websearch, get_weather, newssearch],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

# Create FastAPI app for OpenAI-compatible API endpoint
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Parse request body
    body = await request.json()

    # Extract messages from the request
    messages = body.get("messages", [])

    # Get the user's message (last message in the array)
    user_message = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_message = message.get("content", "")
            break

    if not user_message:
        return Response(
            content=json.dumps({
                "error": {
                    "message": "No user message found in the request",
                    "type": "invalid_request_error"
                }
            }),
            media_type="application/json",
            status_code=400
        )

    # Run the triage agent with the user's message
    result = await Runner.run(triage_agent, user_message)

    # Format the response in OpenAI's chat completion format
    response = {
        "id": f"chatcmpl-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "object": "chat.completion",
        "created": int(datetime.datetime.now().timestamp()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.final_output
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(result.final_output.split()),
            "total_tokens": len(user_message.split()) + len(result.final_output.split())
        }
    }

    return response

# Async function to process user input for Gradio interface
async def process_input(user_input):
    result = await Runner.run(triage_agent, user_input)
    return result.final_output

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Agent System") as interface:
        gr.Markdown("# Agent System Interface")
        gr.Markdown("Enter your query and the system will process it using multiple agents.")

        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question here...",
                    lines=3
                )
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_text = gr.Markdown(label="Response")

        submit_btn.click(
            fn=lambda x: asyncio.run(process_input(x)),
            inputs=input_text,
            outputs=output_text
        )

    return interface

async def main():
    if os.getenv("CLI_MODE", "false").lower() == "true":
        input_text = input("Input: ")
        result = await Runner.run(triage_agent, input_text)
        print(result.final_output)
    else:
        # Start both the API server and Gradio interface
        interface = create_gradio_interface()

        # Launch Gradio in a separate thread
        interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

        # Start FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())
