from flask import Flask, request, jsonify
import cohere
from DiskVectorIndex import DiskVectorIndex
import os

# Set environment variables
os.environ.update({
    "COHERE_API_KEY": "zy8YjaFYCrI1gdeWZyOf2k3NPsKkV2OnjEXtEsKd",
})

co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
index = DiskVectorIndex("Cohere/trec-rag-2024-index")

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint for asking a question and getting a response.

    This endpoint receives a POST request with a JSON payload containing a 'question' field.
    It decomposes the question into sub-questions, searches for relevant documents for each sub-query,
    and generates a response using the COHERE API.

    Returns:
        A JSON response containing the generated response text.

    Raises:
        400: If no question is provided in the request payload.
    """
    # Get the question from the request
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Decompose the question into sub-questions
    prompt = f"Answer the following question with a detailed answer: {question}"
    res = co.chat(
        model="command-r-plus",
        message=prompt,
        search_queries_only=True
    )

    sub_queries = [r.text for r in res.search_queries]

    # Search for relevant documents for each sub-query
    docs = []
    doc_id = 1
    for query in sub_queries:
        hits = index.search(query, top_k=3)
        for hit in hits:
            docs.append({"id": str(doc_id), 'title': hit['doc']['title'], 'snippet': hit['doc']['segment']})
            doc_id += 1

    # Generate the response
    response_text = ""
    for event in co.chat_stream(model="command-r-plus", message=prompt, documents=docs, citation_quality="fast"):
        if event.event_type == "text-generation":
            response_text += event.text
    return jsonify({"response": response_text}), 200

if __name__ == '__main__':
    app.run(debug=True)
