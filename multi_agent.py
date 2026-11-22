from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent


load_dotenv()

llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.5,
        )



def fruit_info(fruit_name: str) -> str:
    """Look up info about a fruit."""
    facts = {
        "tomato": "Botanically a fruit (it develops from a flower and has seeds), "
        "but commonly treated as a vegetable in cooking.",
        "apple": "A pome fruit, related to roses. High in fiber and vitamin C.",
    }
    return facts.get(fruit_name.lower(), f"No specific info found for {fruit_name}.")


def veggie_info(veggie_name: str) -> str:
    """Look up info about a vegetable."""
    facts = {
        "carrot": "A root vegetable, high in beta-carotene (vitamin A).",
        "potato": "A starchy tuber vegetable, a staple food in many cuisines.",
    }
    return facts.get(veggie_name.lower(), f"No specific info found for {veggie_name}.")



fruit_agent = create_agent(
    model=llm,
    tools=[fruit_info],
    system_prompt='You are a fruit expert. Respond in one sentence'

)


vaggie_agent =  create_agent(
    model=llm,
    tools=[veggie_info],
    system_prompt='You are a vaggie expert. Respond in one sentence'

)

def ask_fruit_expert(question:str)->str:
    """Ask the fruit expert. use for All fruit questions."""
    
    response = fruit_agent.invoke({
        'messages':[
            {
                'role':'user','content': question
            }
        ]
    })
    
    return response['messages'][-1].content


def ask_vaggie_expert(question:str)->str:
    """Ask the vaggie expert. use for All fruit questions."""
     
    response = vaggie_agent.invoke({
        'messages':[
            {
                'role':'user','content': question
            }
        ]
    })
    
    return response['messages'][-1].content

main_agent = create_agent(
    model=llm,
    tools=[ask_fruit_expert,ask_vaggie_expert],
    system_prompt='You have two expert: ask_fruit_expert and ask_vaggie_expert. Always delegate questions to appropriate expert.'
)

result = main_agent.invoke({
    'messages':[
        {'role':'user','content':'is a tomato a fruit?'}
    ]
})

print(result['messages'][-1].content)