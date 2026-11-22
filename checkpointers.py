import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

dotenv.load_dotenv()


class Agent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.5,
        )


        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt='Give short answers',
            checkpointer=InMemorySaver()
        )

    def chat_loop(self):
        active_session = 'session_1'
        while True:
            user_input = input(
                f'({active_session}) Ask something (type "q" to terminate or type "ch" to change sessions ): '
            ).strip()

            if user_input.lower() == "q":
                print("Good luck!")
                break
            
            if user_input.lower == 'ch':
                active_session = 'session_2' if active_session == 'session_1' else 'session_1'
                print('Done!')
                continue
                
            
            config = {
                'configurable' : {
                    'thread_id' : active_session
                }
            }


            response = self.agent.invoke({
                "messages": HumanMessage(content=user_input)
            },config=config)


            print(response["messages"][-1].content)


def main():
    agent = Agent()
    agent.chat_loop()


if __name__ == "__main__":
    main()