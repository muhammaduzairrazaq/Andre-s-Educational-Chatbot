# from IPython.display import Markdown, display
# from langchain_core.chat_history import (
#     BaseChatMessageHistory,
#     InMemoryChatMessageHistory,
# )
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import SystemMessage, trim_messages
# from operator import itemgetter
# from langchain_core.runnables import RunnablePassthrough

# import getpass
# import os
# from langchain_core.messages import HumanMessage, AIMessage

# os.environ["OPENAI_API_KEY"] = 'sk-proj-q1GTGj6dEBBxnyx5J9dqT3BlbkFJIOUADKjYN9wyPDa8a3H3'                                                  #getpass.getpass()
# max_token_limits = 3000

# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens = max_token_limits)

# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]


# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# rules = """
# Use simple, easy-to-understand language for elementary school. \
# Introduce technical terms but explain them clearly for middle school. \
# Use a clear, instructive tone with detailed explanations for high school. \
# Always give at least one example even if user did'nt ask for it. \
# Employ a professional tone with technical terms and in-depth explanations for college and advanced levels. \
# Break down complex concepts into smaller, manageable steps for clarity. \
# Engage users conversationally with follow-up questions to gauge understanding and encourage further discussion. \
# Adapt to user feedback, modifying responses based on additional questions to maintain an interactive learning experience. \
# Exhibit patience and empathy, especially with users struggling with concepts. \
# Reassure them and encourage them to ask for clarification. \
# Be mindful of learning styles and cultural differences, adapting responses to accommodate different learning paces and styles. \
# Be respectful of diverse backgrounds and cultural contexts. \
# Use Markdown for educational responses to structure content clearly (e.g., Introduction, Explanation, Examples, Summary) to make information more organized and accessible.
#   """

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             f"You are a helpful academic assistant, if the user greets you then greet him back and if the user is asking a educatinal query then respond him back strictly following the rules {rules}. Always stick to the educational domain.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = prompt | model

# # Adding message history
# with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# # Setting up trimmer
# trimmer = trim_messages(
#     max_tokens=65,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

# messages = [
#     SystemMessage(content="You are a helpful academic assistant"),
# ]

# ### Adding trimer to the message for trimming old messages from the context
# chain = (
#     RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
#     | prompt
#     | model
# )

# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="messages",
# )

# config = {"configurable": {"session_id": "1a"}} # change the session id for changing the context

# # Function to generate llm response
# def invoke_llm(query):
#     response = with_message_history.invoke(
#         {
#             "messages": messages + [HumanMessage(content=query)],
#             "language": "English"
#         },
#         config=config,
#     )
#     messages.append(HumanMessage(content=query)) # Saving user message for context
#     messages.append(AIMessage(content=query)) # Saving ai message for context
#     return response.content

# import panel as pn
# from IPython.display import Markdown

# pn.extension()

# # Define the function to handle button click
# def on_button_click(event):
#     query = inp.value  
#     if query.strip():
#         user_query = f'**Query:** {query} \n\n'
#         output_area.object = user_query
#         response = invoke_llm(query)
#         # Display the response as Markdown
#         output_area.object = user_query + response

# # Create widgets
# inp = pn.widgets.TextInput(value="", placeholder='Ask me anything...', width=500, height=40,)
# button_conversation = pn.widgets.Button(name="Chat!", width=90, height=40, button_type='primary')

# # Bind the button click event to the handler function
# button_conversation.on_click(on_button_click)

# # Create an output area for displaying the Markdown response
# output_area = pn.pane.Markdown(align='center', width=600, height=500)

# # Arrange widgets and output area in a layout
# input_row = pn.Row(inp, button_conversation, align='center')

# dashboard = pn.Column(
#     input_row,
#     output_area,
#     align='center',
#     sizing_mode='stretch_width',
# )


# # Create a final layout
# final_dashboard = pn.Column(
#     dashboard,
#     sizing_mode='stretch_width',
#     align='center'
# )

# # Display the dashboard
# final_dashboard.show()




from IPython.display import Markdown, display
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

import getpass
import os
from langchain_core.messages import HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = 'sk-proj-q1GTGj6dEBBxnyx5J9dqT3BlbkFJIOUADKjYN9wyPDa8a3H3'  # getpass.getpass()
max_token_limits = 3000

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=max_token_limits)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)

rules = """
Use simple, easy-to-understand language for elementary school. \
Introduce technical terms but explain them clearly for middle school. \
Use a clear, instructive tone with detailed explanations for high school. \
Always give at least one example even if user did'nt ask for it. \
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
            f"You are a helpful academic assistant, if the user greets you then greet him back and if the user is asking a educational query then respond him back strictly following the rules {rules}. Always stick to the educational domain.",
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

### Adding trimer to the message for trimming old messages from the context
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

config = {"configurable": {"session_id": "1a"}}  # change the session id for changing the context

# Function to generate llm response
def invoke_llm(query):
    response = with_message_history.invoke(
        {
            "messages": messages + [HumanMessage(content=query)],
            "language": "English"
        },
        config=config,
    )
    messages.append(HumanMessage(content=query))  # Saving user message for context
    messages.append(AIMessage(content=query))  # Saving ai message for context
    return response.content

import panel as pn
from IPython.display import Markdown

pn.extension()

# Define the function to handle button click
def on_button_click(event):
    query = inp.value  
    if query.strip():
        user_query = f'**Query:** {query} \n\n'
        output_area.object = user_query
        response = invoke_llm(query)
        # Display the response as Markdown
        output_area.object = user_query + response

# Create widgets
inp = pn.widgets.TextInput(value="", placeholder='Ask me anything...', width=500, height=40,)
button_conversation = pn.widgets.Button(name="Chat!", width=90, height=40, button_type='primary')

# Bind the button click event to the handler function
button_conversation.on_click(on_button_click)

# Create an output area for displaying the Markdown response
output_area = pn.pane.Markdown(align='center', width=600, height=500)

# Arrange widgets and output area in a layout
input_row = pn.Row(inp, button_conversation, align='center')

dashboard = pn.Column(
    input_row,
    output_area,
    align='center',
    sizing_mode='stretch_width',
)

# Create a final layout
final_dashboard = pn.Column(
    dashboard,
    sizing_mode='stretch_width',
    align='center'
)

# Serve the dashboard on port 5001
pn.serve(final_dashboard, port=5001, show=True)
