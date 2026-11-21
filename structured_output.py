import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder  
from pydantic import BaseModel, Field 

dotenv.load_dotenv()

class User(BaseModel):
    name:str = Field(description='The name of the user')
    age:int = Field(description='The age of the user')
    companies:list[str] = Field(description='the companies associated of the user')
    net_worth:str = Field(description='the net worth of the users')

class Agent:
    def __init__(self):
         self.llm = ChatGroq( 
                    model="openai/gpt-oss-120b",
                    temperature=0.1
                )
         
    def chat(self):
        structured_llm = self.llm.with_structured_output(User,method='json_schema' )
        response = structured_llm.invoke('''Elon Reeve Musk (/ˈiːlɒn/ ⓘ; born June 28, 1971) is an industrialist, futurist, and former United States public official. He is the chief executive officer and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire in history (in terms of US dollars) in 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion.

Musk was born into the wealthy Musk family in Pretoria, South Africa. He emigrated to Canada in 1989; he has Canadian citizenship as his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California, United States, to pursue business ventures. In 1995, he co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

''')    
        print(response)
        
    
def main():
    agent = Agent()
    agent.chat()
    
    
if __name__ == "__main__":
    main()