from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-70b-8192"
)

print(llm.invoke("Say hello in one sentence"))
