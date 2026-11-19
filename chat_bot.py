from dotenv import load_dotenv
from openai import OpenAI
import os
import time
import tiktoken
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
        
        
    def trim_messages(self,max_number = 4):
        if len(self.messages) < max_number:
            return
        system_messages = [m for m in self.messages if m['role']=='system']
        other_messages = [m for m in self.messages if m['role']!='system']
        trimed_messages = other_messages[-max_number:]
        self.messages = system_messages + trimed_messages
            
    def bot(self,message:str):
        self.messages.append({
                        'role':"user",
                        'content':message
                        })
        response = self.llm.chat.completions.create(model='openai/gpt-oss-120b',
                                                messages=self.messages,
                                                stream= True)

        response_text =  ''
        
        for chunk in response:
            if chunk.choices:
                 delta = chunk.choices[0].delta.content
                 if delta:
                     print(delta,end='',flush=True)
                     response_text += delta
                     time.sleep(0.05)
                     
            if chunk.usage:
                self.total_tokens += chunk.usage.total_tokens
                self.input_tokens += chunk.usage.prompt_tokens
                self.output_tokens += chunk.usage.completion_tokens
                

        self.messages.append({
            'role':'assistant',
            'content':response_text
        })

        self.trim_messages()

        return response_text

    def chat(self):
        encoding = tiktoken.get_encoding('cl100k_base')
        while True:
            user_input = input('(Type "q" to quit) Ask Something : ').strip().lower()
            if user_input == 'q':
                print('Good luck')
                break
            token_count = len(encoding.encode(user_input))
            
            if token_count > 100:                                                     # limit User Prompt
                print('Your prompt exceeded the limit. Try a shorter one, please')
                continue
            self.bot(user_input)

    def cost(self):
        print("Cost : $",(self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1_000_000)

def main():
    
    chat = ChatBot()
    chat.chat()
    print(f"Consumed tokens: {chat.total_tokens}")
    chat.cost()





if __name__ == '__main__':
    main()