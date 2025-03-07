from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType

from langchain.tools import Tool

from langchain_community.vectorstores import FAISS


embeddings = OllamaLLM(model="multi-qa-MiniLM-L6-cos-v1")  

# تحميل بيانات FAISS
faiss_index = FAISS.load_local("faiss_index", embeddings)

faiss_tool = Tool(
    name="FAISS_Search",
    func=lambda query: faiss_index.similarity_search(query, k=5),
    description="بحث في قاعدة بيانات FAISS باستخدام استعلام نصي."
)

# إعداد LLM باستخدام Ollama
llm = OllamaLLM(model="llama3.2")

# تهيئة الوكيل بالأدوات المتاحة
agent = initialize_agent(
    tools=[faiss_tool],  # إضافة أداة البحث
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# تنفيذ البحث
response = agent.run("What is the capital of France?")
print(response)
