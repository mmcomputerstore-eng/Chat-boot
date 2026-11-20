import hashlib
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


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
        
        for i,chunk in enumerate(retreved_chunks):
            print(f'---- chunk {i+1} ----\n')
            print(f'Score : {chunk[1]:.3f}')
            print(f'Metadata : source : {chunk[0].metadata['source']}')
            print(f'Page : source : {chunk[0].metadata['page']}')
            print(f'{chunk[0].page_content[:200]}\n')
            
            
    





if __name__ == '__main__':
    main()