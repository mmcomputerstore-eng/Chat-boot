import hashlib
import dotenv
import os 
from openai import OpenAI
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
dotenv.load_dotenv()

client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs = {'normalize_embeddings':True}
)

vactor_store = Chroma(
    collection_name = 'medical_data',
    embedding_function=model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space':'cosine'}
)

def llm(query,context):
    system_prompt = f"""You are a medical information assistant. Answer the user's question using ONLY the provided source documents.
        Rules you must follow:

        1. Base your answer ONLY on the provided sources. Do not use outside Knowledge.
        2. If the sources   don't contain enough information to answer, say exactly:
        "I don't have enough information to answer this question."
        3. Cite sources using [Source N] notation whenever you use information from them.
        4. If sources contradice each other, mention the contradiction.
        5. Be consice. Answer in 3-5 sentences unless the question requires more detail.
        6. Never speculate or make assumptions beyond what the sources say. 

        Sources:
        {context}"""
    
    messages = [
        {
            'role':'system','content':system_prompt
        },
        {
            'role':'user','content':query
        }
    ]
    
    response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0.7,
            messages=messages
        )
    
    return response.choices[0].message.content
    


    

def generate_id(docs):
    return [
        hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs
    ]

def read_pdf_file(file_name):
    docs = []
    
    reader = PdfReader(file_name)
    
    for i , page in enumerate(reader.pages):
        doc = Document(page_content=page.extract_text() or "", metadata= {'source':file_name,'page':i+1})
        docs.append(doc)
        
    return(docs)





def main():
    docs = read_pdf_file('/home/mubeen/Documents/genrative AI/Chat boot/symptoms.pdf')
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 110,
        chunk_overlap = 10
    )
    chunks = splitter.split_documents(docs)
    
    ids = generate_id(chunks)
    
    vactor_store.add_documents(chunks,ids=ids)
     

    while True:
        user_input = input('(Type "q" to quit) Ask Something : ').strip().lower()
    
        # Exit condition
        if user_input == "q":
            print("Good luck")
            break
        
        retreved_chunks = vactor_store.similarity_search_with_score(user_input, k=3)
        
        parts = []
        
        for i,(doc,score) in enumerate(retreved_chunks):
            parts.append(f"[source : {i+1} | Page {doc.metadata.get('page','?')} | Retrival score {score:.3f} \n {doc.page_content}]")
            
            
        context = '\n\n-----\n\n'.join(parts)
        
            
        
        print(llm(query=user_input,context=context))
            




if __name__ == '__main__':
    main()