from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Groq model
groq_model = Groq(id="llama-3.2-11b-vision-preview")

# Define the Finance Agent using YFinanceTools
finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
            company_info=True
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

# Define the Web Search Agent using DuckDuckGo
web_search_agent = Agent(
    name='Web Search Agent',
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include source"],
    show_tools_calls=True,
    markdown=True
)

# Multi-agent setup: Combine finance and web search agents
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=groq_model,
    instructions=["Always include source and link", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[multi_ai_agent,web_search_agent,finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)