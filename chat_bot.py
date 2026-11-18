from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()


class ChatBot:
    def __init__(self):
        self.llm = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url='https://api.groq.com/openai/v1'
            )
        self.messages = [
                    {
                        'role':'system',
                        'content':'''you are a helpful assistant. your name is Neon_bot. Introduce yourself in the first message.
                        give short and helpful answers'''
                    }
            ]

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        
    
    def bot(self,message:str):
        self.messages.append({
                        'role':"user",
                        'content':message
                        })
        response = self.llm.chat.completions.create(model='openai/gpt-oss-120b',
                                                messages=self.messages)

        response_text =  response.choices[0].message.content

        self.messages.append({
            'role':'assistant',
            'content':response_text
        })
        self.total_tokens += response.usage.total_tokens
        self.input_tokens += response.usage.prompt_tokens
        self.output_tokens += response.usage.completion_tokens


        return response_text

    def chat(self):
        while True:
            user_input = input('(Type "q" to quit) Ask Something : ').strip().lower()
            if user_input == 'q':
                print('Good luck')
                break
            bot_answer = self.bot(user_input)
            print(bot_answer)

    def cost(self):
        print("Cost : $",(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000)

def main():
    
    chat = ChatBot()
    chat.chat()
    print(f"Consumed tokens: {chat.total_tokens}")
    chat.cost()





if __name__ == '__main__':
    main()