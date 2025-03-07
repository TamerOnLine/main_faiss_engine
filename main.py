from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool

def scrape_webpage(url: str) -> str:

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text_content = "\n".join([p.get_text() for p in paragraphs if p.get_text()])

    return text_content[:5000] + "..." if len(text_content) > 5000 else text_content
    
web_scraper_tool = Tool(
    name="WebScraper",
    func=scrape_webpage,
    description="Scrapes webpage content and extracts text data.",
    return_direct=True,
)

llm = OllamaLLM(model="llama3.2")
   
agent = initialize_agent(
    tools=[web_scraper_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    allowed_tools=[],
    handle_parsing_errors=True,
    )  


def process(query):

    print(f"Processing query: {query}")
    result = agent.invoke(query)
    response = result.get("output", "No response")
    print(f"Response: {response}")
    return response