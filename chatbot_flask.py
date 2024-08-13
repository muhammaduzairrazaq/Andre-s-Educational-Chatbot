from flask import Flask, request, jsonify, render_template
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
import os
from markdown import markdown

os.environ["OPENAI_API_KEY"] = 'sk-proj-q1GTGj6dEBBxnyx5J9dqT3BlbkFJIOUADKjYN9wyPDa8a3H3'
config = {"configurable": {"session_id": "oneA"}} # change the session id for changing the context

max_token_limits = 3000
temperature = 0.3
from langchain_openai import ChatOpenAI

app = Flask(__name__)

model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=max_token_limits, temperature=temperature)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rules = """
Use simple, easy-to-understand language for elementary school. \
Introduce technical terms but explain them clearly for middle school. \
Use a clear, instructive tone with detailed explanations for high school. \
Always give at least one example even if user didn't ask for it. \
Employ a professional tone with technical terms and in-depth explanations for college and advanced levels. \
Break down complex concepts into smaller, manageable steps for clarity. \
Engage users conversationally with follow-up questions to gauge understanding and encourage further discussion. \
Adapt to user feedback, modifying responses based on additional questions to maintain an interactive learning experience. \
Exhibit patience and empathy, especially with users struggling with concepts. \
Reassure them and encourage them to ask for clarification. \
Be mindful of learning styles and cultural differences, adapting responses to accommodate different learning paces and styles. \
Be respectful of diverse backgrounds and cultural contexts. \
Use Markdown for educational responses to structure content clearly (e.g., Introduction, Explanation, Examples, Summary) to make information more organized and accessible.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a helpful academic assistant. If the user greets you, greet them back. If the user asks an educational query, respond strictly following the rules: {rules}. Always stick to the educational domain.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# Adding message history
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# Setting up trimmer
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="You are a helpful academic assistant"),
]

# Adding trimmer to the message for trimming old messages from the context
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "1a"}}  # Change the session ID for changing the context

# Function to generate LLM response
def invoke_llm(query):
    # print(f"Invoking LLM with query: {query}")
    response = with_message_history.invoke(
        {
            "messages": messages + [HumanMessage(content=query)],
            "language": "English"
        },
        config=config,
    )
    messages.append(HumanMessage(content=query))  # Saving user message for context
    messages.append(AIMessage(content=response.content))  # Saving AI message for context
    # print(f"Received response: {response.content}")
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    # print(f"Received query: {query}")
    if query:
        # response = markdown(invoke_llm(query))
        response = invoke_llm(query)
        return jsonify({'response': response})
    print("No query provided")
    return jsonify({'error': 'No query provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)