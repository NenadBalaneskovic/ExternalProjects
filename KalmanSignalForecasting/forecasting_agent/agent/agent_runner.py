# agent/agent_runner.py
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from models.local_llm import load_local_llm
from agent.tools import forecast_asset  # Use the version from tools.py


def create_agent():
    llm = load_local_llm()
    tools = [Tool.from_function(forecast_asset)]
    agent = initialize_agent(tools, llm,
                             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    return agent
