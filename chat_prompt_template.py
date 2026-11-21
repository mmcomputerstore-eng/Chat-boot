import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder   

dotenv.load_dotenv()

class Agent:
    def __init__(self):
         self.llm = ChatGroq( 
                    model="openai/gpt-oss-120b",
                    temperature=0.5
                )
         
    def chat(self):
        
        history = [
            HumanMessage('I am tired'),
            AIMessage('>>>Ich bin mude<<<')
            HumanMessage('I hate you'),
            AIMessage('>>>Ich hasse dich<<<')
            
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ('system','you are a translator from {origin} to {to}'),
            MessagesPlaceholder(variable_name='conversation_history'),
            ('human','{input}')
        ])
        
        
        chain = prompt | self.llm
        
        response = chain.invoke({
            'conversation_history':history,
            'origin' : 'English',
            'to' :'German',
            'input' : 'I love you'
        })
        
        print(response.content)
        
        
        
    
def main():
    agent = Agent()
    agent.chat()
    
    
if __name__ == "__main__":
    main()