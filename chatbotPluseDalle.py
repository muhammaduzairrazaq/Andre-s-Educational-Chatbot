import os
from io import BytesIO
import openai
import base64
import requests
from PIL import Image
import tkinter as tk
from PIL import ImageTk
from flask import Flask, request, jsonify, render_template, Response
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import seaborn as sns

import io
import re


# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = 'sk-proj-q1GTGj6dEBBxnyx5J9dqT3BlbkFJIOUADKjYN9wyPDa8a3H3'

# Create the chat model
max_token_limits = 3000
temperature = 0.3
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4", max_tokens=max_token_limits, temperature=temperature)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

rules = """
Employ a professional tone with technical terms and in-depth explanations for college and advanced level queries. \
Dont Draw any animations just explain the topic asked by user like if the user asked you to draw a plot and there is nothing to explain you can simply explain the analysis on the graph. \
Break down complex concepts into smaller, manageable steps for clarity. \
Always provide examples while responding to user queries. \
Be respectful of diverse backgrounds and cultural contexts. \
Use Markdown for educational responses to structure content clearly (e.g., Introduction, Explanation, Examples, Summary) to make information more organized and accessible. \
Dont say anything like this 'As an AI text-based model, I'm unable to draw graphs or animations' because the graphs or animations will be displyed to user by external systems so simple ignore it. \
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a helpful academic assistant having deep understanding of educational concepts, always respond in a friendly tone by following the rules {rules}. Always stick to the educational domain.",
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

config = {"configurable": {"session_id": "5a"}}  # Change the session ID for changing the context

def invoke_llm(query):
    response = with_message_history.invoke(
        {
            "messages": messages + [HumanMessage(content=query)],
            "language": "English"
        },
        config=config,
    )
    messages.append(HumanMessage(content=query))  # Saving user message for context
    messages.append(AIMessage(content=response.content))  # Saving AI message for context
    return response.content

def analyze_query(query):
    max_token_limits = 1000
    temperature = 0.3
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=max_token_limits, temperature=temperature)

    message = f"""
    You are a chatbot that provides assistance with educational queries. Based on the user's query, you need to determine whether a plot or animation is suitable. Here's how you should respond:

    1. If the query explicitly asks for a plot or animation (e.g., "plot a graph," "show an animation," etc.) and involves a mathematical function, equation, or concept that can be visualized effectively with a plot: Return the Python code that can be used to draw the graph. Use libraries such as Matplotlib, Seaborn etc, but do not add 'plt.show()'.

    2. If the query asks for an animation, but a static plot is more appropriate due to simplicity, clarity, or accuracy: Generate and return the code for the plot instead.

    3. If the query explicitly asks for a plot or animation, but neither is suitable: Return ANIMATION.

    4. If the query does not explicitly ask for a plot or animation or any other visual illustration: Return FALSE.

    5. User can also ask you to draw different charts (line, bar, histogram, pie, scatter, area, etc): Return the Python code that can be used to draw the graph. Use libraries such as Matplotlib or Seaborn, but do not add 'plt.show()'.

    Critical points:
    
    -Dont use backticks ` in your response
    -Python code will be executed in a dynamic environment, such as a web server, where the code might be executed dynamically via exec() or similar methods. Ensure that any variables or objects needed within callback functions or loops are explicitly passed or encapsulated to prevent NameError or scope issues. Avoid relying on global variables.
    -The generated code should only use following libraries matplotlib seaborn pandas plotly altair squarify
    Here are some examples:

    Query: Plot the function \( f(x) = x^2 - 4x + 4 \) and identify its vertex and axis of symmetry.
    Response:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-1, 5, 400)
    y = x**2 - 4*x + 4

    plt.plot(x, y, label='$f(x) = x^2 - 4x + 4$')
    plt.title('Plot of $f(x) = x^2 - 4x + 4$')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()

    Query: Compare the sine and cosine functions using animations.
    Response:
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 1000)

    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sin, label='Sine Function', color='blue')
    plt.plot(x, y_cos, label='Cosine Function', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Sine and Cosine Functions')
    plt.legend()
    plt.grid(True)

    Query: What is sine function explain with animation.
    Response:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 2*np.pi, 100)
    sine = np.sin(x)

    plt.plot(x, sine, label='Sine function')
    plt.title('Sine Function')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)

    Query: How does the human respiratory system function.
    Response: FALSE

    Query: Show an illustration of the sigmoid function.
    Response:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

    x = np.linspace(-10, 10, 100)
    fig, ax = plt.subplots()
    line, = ax.plot(x, 1 / (1 + np.exp(-x)))

    def update(frame, line, x):
        line.set_ydata(1 / (1 + np.exp(-(x + frame / 10))))
        return line,

    ani = FuncAnimation(fig, update, frames=100, fargs=(line,x), blit=True)

    plt.title('Sigmoid Function Animation')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.grid(True)

    Query: What is the time complexity of quicksort show an animation.
    Response: ANIMATION

    Query: Draw a graph of human heart.
    Response: ANIMATION

    Query: Graph the relationship between the number of hours studied and exam scores.
    Response:
    import matplotlib.pyplot as plt

    hours = [1, 2, 3, 4, 5]
    scores = [55, 60, 65, 70, 75]

    plt.scatter(hours, scores)
    plt.plot(hours, scores, linestyle='dashed', color='red')
    plt.title('Hours Studied vs Exam Scores')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Scores')
    plt.grid(True)

    Query: {query}
    Response:
    """

    result = model.invoke(message)
    return parser.invoke(result)

def plot(code_string):

    # Use the Agg backend to avoid gui related issues
    plt.switch_backend('Agg')

    # Create a buffer to hold the image
    buf = io.BytesIO()

    # Execute the code
    exec(code_string)

    # Save the figure to the buffer
    plt.savefig(buf, format='png')

    # Close the plot to avoid display issues
    plt.close()

    # Move the buffer cursor to the beginning
    buf.seek(0)

     # Encode the image in base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_base64

def generate_image(prompt):
    from openai import OpenAI
    client = OpenAI(api_key="sk-proj-q1GTGj6dEBBxnyx5J9dqT3BlbkFJIOUADKjYN9wyPDa8a3H3")

    
    image_params = {
        "model": "dall-e-3", 
        "n": 1,
        "size": "1024x1024",
        "prompt": prompt,
        "response_format": "url"  # Use "b64_json" if you prefer base64
    }
    
    try:
        images_response = client.images.generate(**image_params)
    except openai.APIConnectionError as e:
        return {"error": f"Server connection error: {e.__cause__}"}
    except openai.RateLimitError as e:
        return {"error": f"OpenAI RATE LIMIT error {e.status_code}: {e.response}"}
    except openai.APIStatusError as e:
        return {"error": f"OpenAI STATUS error {e.status_code}: {e.response}"}
    except openai.BadRequestError as e:
        return {"error": f"OpenAI BAD REQUEST error {e.status_code}: {e.response}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
    
    image_url_list = [image.model_dump()["url"] for image in images_response.data]
    if image_url_list:
        return {"image_url": image_url_list[0]}  # Return the first image URL

    return {"error": "No image data was obtained."}


def remove_related_words(query):
    # Convert the query to lowercase
    query = query.lower()
    
    # List of words and their plural forms related to graphics to remove
    words_to_remove = [
        'animation', 'animations', 'plot', 'plots', 'graph', 'graphs',
        'draw', 'draws', 'illustration', 'illustrations', 'chart', 'charts',
        'diagram', 'diagrams', 'figure', 'figures', 'image', 'images',
        'visual', 'visuals', 'sketch', 'sketches', 'render', 'renders'
    ]
    
    # Create a regex pattern that matches any of the words
    pattern = r'\b(?:' + '|'.join(words_to_remove) + r')\b'
    
    # Use re.sub to replace these words with an empty string
    cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
    
    # Return the cleaned query
    return ' '.join(cleaned_query.split())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    if query:
        # Analyze the query
        result = analyze_query(query)

        # Debug
        print(f'Result of the analyze report {result}')

        image_response = None
        plot_base64 = None


        if result == 'ANIMATION':
            # Generate image
            image_response = generate_image(query+' without adding any textual labels')
        
        if result != 'ANIMATION' and result !=  'FALSE':
            plot_base64 = plot(result)  

        # Removing the keywords related to animation plotting etc
        # Debug
        print(f'Query before removing unreleated words {query}')
        # query = remove_related_words(query) 
        # query = "" + query
        # Debug
        print(f'Query after removing unreleated words {query}')
        # Get text response
        text_response = invoke_llm(query)
        
        response = {
            'response': text_response,
            'image': image_response.get('image_url') if image_response else None,
            'plot': plot_base64,
            'error': image_response.get('error') if image_response else None
        }
        
        return jsonify(response)
    return jsonify({'error': 'No query provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
