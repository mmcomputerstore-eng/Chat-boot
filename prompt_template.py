import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()

class Agent:
    def __init__(self):
         self.llm = ChatGroq( 
                    model="openai/gpt-oss-120b",
                    temperature=0.5
                )
         
    def chat(self):
        template = '''Dear {user_name}
        Apologizing for the bad experince, we will set an appointment to 
        further discuss your problem.
        
        Sincerely
        {brand_name}
        '''
        
        prompt = PromptTemplate(input_variables=['user_name','brand_name'],template=template)
        result = prompt.format(user_name='mubeen',   brand_name='MM Computer Store')
        
        response = self.llm.invoke('hello')
        print(response.content)
        
        
    
def main():
    agent = Agent()
    agent.chat()
    
    
if __name__ == "__main__":
    main()