import os
import sys
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams, StdioServerParameters
from google.genai import types

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
load_dotenv(os.path.join(root_dir, ".env"), override=True)
load_dotenv(override=True)

# Resolve absolute path to the local mcp_server.py file
base_dir = os.path.dirname(os.path.abspath(__file__))
mcp_server_path = os.path.join(base_dir, "mcp_server.py")

# Use current python executable to start the stdio server
python_executable = sys.executable or "python"

# Initialize connection parameters with StdioConnectionParams as recommended
server_params = StdioServerParameters(
    command=python_executable,
    args=[mcp_server_path]
)
connection_params = StdioConnectionParams(
    server_params=server_params
)

# Create McpToolset instance
mcp_toolset = McpToolset(connection_params=connection_params)

# System instruction for the agent
system_instruction = (
    "You are a GitHub profile analyst and dev card generator. When a user gives you a GitHub username, "
    "you ALWAYS follow this exact sequence: first call scrape_github, then analyze_profile with the result, "
    "then generate_card_html with all three inputs, then save_card. Never skip steps. "
    "Be enthusiastic about developers' work. If the profile is private or doesn't exist, say so clearly."
)

# Export the agent as github_card_agent using Gemini 2.5 Flash
github_card_agent = LlmAgent(
    name="github_card_agent",
    model="gemini-2.5-flash",
    instruction=system_instruction,
    tools=[mcp_toolset]
)


class GitHubCardAgent:
    """
    Orchestrates the GitHub Developer Card Generation pipeline
    using Google ADK and local MCP tools.
    """
    def __init__(self):
        self.agent = github_card_agent
        self.toolset = mcp_toolset
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="github_dev_card_app",
            auto_create_session=True
        )

    async def initialize(self):
        # Tools are registered synchronously in recent ADK versions
        pass

    async def generate_card(self, username: str) -> str:
        prompt = f"Generate and save a developer card for the GitHub user: {username}"
        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
        response_texts = []
        async for event in self.runner.run_async(
            user_id="default_user",
            session_id=username,
            new_message=message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        response_texts.append(part.text)
        return "".join(response_texts)

    async def close(self):
        try:
            await self.toolset.close()
        except Exception as e:
            print(f"Error closing McpToolset: {e}", file=sys.stderr)
