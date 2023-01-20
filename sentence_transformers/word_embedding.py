from sentence_transformers import SentenceTransformer

model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

def word_embedding(event, context):
    return model.encode(event['text'])
