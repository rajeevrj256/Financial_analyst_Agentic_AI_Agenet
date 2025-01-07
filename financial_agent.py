from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini



# Initialize the model
model=Gemini(id='gemini-1.5-flash-exp-0827', params=None, client=None)  


# Define the Finance Agent using YFinanceTools
finance_agent = Agent(
    name="Finance AI Agent",
    model=model,
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
    model=model,
    tools=[DuckDuckGo()],
    instructions=["Always include source"],
    show_tools_calls=True,
    markdown=True
)

# Multi-agent setup: Combine finance and web search agents
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Gemini(id="gemini-1.5-flash-exp-0827"),
    instructions=["Always include source", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

# Example query for the Multi AI Agent

query = input("How I help You?")
messages = [{"role": "user", "content": query}]
print("Message to send:", query)
print("Stream flag:", True)

# Execute the query and print the response
multi_ai_agent.print_response(query, stream=True)
