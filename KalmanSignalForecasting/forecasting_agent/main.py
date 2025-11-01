# main.py
from agent.agent_runner import create_agent

if __name__ == "__main__":
    agent = create_agent()
    prompt = "Forecast Asset_2 with quantum denoising and plot the result"
    response = agent.run(prompt)
    print(response)
