import dotenv
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def find_closest(query, sentences):
    """
    Finds the sentence in `sentences` that has the highest cosine similarity to `query`.
    """
    if not sentences:
        return None

    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)
    
    closest_index = 0
    closest_score = -1.0
    
    for index, vector in enumerate(embeddings):
        # Compute cosine similarity between sentence vector and query embedding
        similarity = util.cos_sim(vector, query_embedding).item()
        
        if similarity > closest_score:
            closest_score = similarity
            closest_index = index

    return sentences[closest_index]


def main():
    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]
    
    query = 'i am feeling great today'
    
    
    embeddings = model.encode(sentences)
    query_embedding = model.encode(query)
    
    result = util.semantic_search(query_embedding, embeddings, top_k=2)
    
    # result[0] contains the top_k matches for the first query
    print('Semantic Search : ', sentences[result[0][0]['corpus_id']])
    print('Semantic Search : ', sentences[result[0][1]['corpus_id']])
    
    closest_sentence = find_closest(query, sentences)
    print("Closest sentence:", closest_sentence)


if __name__ == "__main__":
    main()
