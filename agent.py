import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

dotenv.load_dotenv()


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: name of the city
    """
    print(f"[Tool: get_weather called for '{city}']")
    return {
        'city': city,
        'weather': 'sunny, 22C'
    }


class Agent:
    def __init__(self):
        self.llm = ChatGroq( 
            model="openai/gpt-oss-120b",
            temperature=0.5
        )
        self.messages = [
            SystemMessage('Give short answers. Use the provided tools whenever needed.')
        ]
        
        self.agent = create_agent(
            model=self.llm,
            tools=[get_weather]
        )
        
    def chat_loop(self):
        while True:
            user_input = input('(Type "q" to quit) Ask Something : ').strip()

            # Exit condition
            if user_input.lower() == "q":
                print("Good luck")
                break
            
            if not user_input:
                continue
            
            self.messages.append(HumanMessage(user_input))
            
            response = self.agent.invoke({'messages': self.messages})

            # Update conversation history with all messages (including intermediate tool calls and tool responses)
            self.messages = response['messages']
            
            print(self.messages[-1].content)


def main():
    agent = Agent()
    agent.chat_loop()


if __name__ == '__main__':
    main()