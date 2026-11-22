import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware

dotenv.load_dotenv()


class Agent:
    def __init__(self, checkpointer):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.5,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="Give short answers",
            checkpointer=checkpointer,
            middleware=[SummarizationMiddleware(
                model=self.llm,
                trigger=('tokens',100),
                keep=('messages',5)
            )]
        )

    def chat_loop(self):
        active_session = 'session_3'
        while True:
            user_input = input(
                f'({active_session}) Ask something (type "q" to terminate): '
            ).strip()

            # Exit condition
            if user_input.lower() == "q":
                print("Good luck!")
                break

           
            # Ignore empty inputs
            if not user_input:
                continue

            config = {
                'configurable': {
                    'thread_id': active_session
                }
            }

            # Pass message as a list of messages
            response = self.agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)]
                },
                config=config
            )

            print(response["messages"][-1].content)

        # read old messages from database
        config_history = {
                    'configurable': {
                    'thread_id': 'session_3'
                    }
                }
        
        state = self.agent.get_state(config_history)
        print(state.values['messages'])
        

def main():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL is not set. Please add it to your .env file.")

    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()
        agent = Agent(checkpointer)
        agent.chat_loop()


if __name__ == "__main__":
    main()