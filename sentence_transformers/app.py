from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./model')

def lambda_handler(event, context):
    if isinstance(event['text'], str):
        return {
            'statusCode': 200,
            'embeddings': model.encode([event['text']]).tolist()
            }
    elif isinstance(event['text'], list):
        return {
            'statusCode': 200,
            'embeddings': model.encode(event['text']).tolist()
            }
    else:
        return {
            'statusCode': 200,
            'embeddings': []
            }
