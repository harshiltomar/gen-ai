from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

model.invoke("What is the capital of India", temperature=0.0, max_completion_tokens=10)
## Low temperature means the model will be more deterministic and less creative
## High temperature means the model will be more creative and less deterministic

result = model.invoke("What is the captial of India")

print(result.content)