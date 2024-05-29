from flask import Flask, request, jsonify
import requests
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

def api_call():
    url = "https://general-runtime.voiceflow.com/knowledge-base/query"
    payload = {
        "question": "Who is Ian?",
        "chunkLimit": 2,
        "synthesis": True,
        "settings": {
            "model": "claude-instant-v1",
            "temperature": 0.1,
            "system": "You are an AI FAQ assistant. Information will be provided to help answer the user's questions. Always summarize your response to be as brief as possible and be extremely concise. Your responses should be fewer than a couple of sentences. Do not reference the material provided in your response."
        }
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "VF.DM.65c5cd65a0c117f719c36177.NXM1ocQfxv39x7kX"
    }

    response = requests.post(url, json=payload, headers=headers)
    api_response = response.json()

    # Extract content and format it
    formatted_content = []
    for index, chunk in enumerate(api_response['chunks'], start=1):
        content = chunk['content']
        formatted_content.append(f"Reference {index}: {content}")

    # Extract the response content from the API response
    user_question = "Who is Ian?"

    prompt = PromptTemplate(
        input_variables = ["formatted_content", "user_question"],
        template = '''Based on this reference below: 
        
        {formatted_content}

        Answer the user question: {user_question}
        '''
    )

    # OpenAI API
    chatopenai = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = "")
    llmchain_chat = LLMChain(llm = chatopenai, prompt = prompt)
    answer = llmchain_chat.invoke({"formatted_content": formatted_content, "user_question": user_question})
    answer = answer['text']

    return jsonify({"chunks": formatted_content, "answer": answer})

api_call()