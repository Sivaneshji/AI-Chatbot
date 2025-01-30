from flask import Flask, render_template, request, jsonify, session
import os
import logging
from dotenv import load_dotenv
from groq import Groq
from flask_session import Session
from convex import ConvexClient  # Import Convex

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Retrieve API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
CONVEX_URL = os.getenv('CONVEX_URL')  # Get your Convex deployment URL from .env

# Validate API Key
if not GROQ_API_KEY:
    logging.critical("API key for Groq is missing. Please set the GROQ_API_KEY in the .env file.")
    exit(1)

if not CONVEX_URL:
    logging.critical("Convex URL is missing. Please set the CONVEX_URL in the .env file.")
    exit(1)

# Initialize Flask app
app = Flask(__name__)

# Configure Flask session
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "supersecretkey"  # Change this for production security
Session(app)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize Convex client
convex = ConvexClient(CONVEX_URL)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_input = request.args.get('msg')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Retrieve or initialize conversation history in session
    if 'conversation' not in session:
        session['conversation'] = [{"role": "system", "content": "You are a helpful AI assistant."}]

    # Append user input to conversation history
    session['conversation'].append({"role": "user", "content": user_input})

    try:
        response_text = get_groq_response(session['conversation'])

        # Append AI response to conversation history
        session['conversation'].append({"role": "assistant", "content": response_text})

        # Store chat in Convex
        store_chat_in_convex(user_input, response_text)

        return jsonify({"response": response_text})
    except Exception as e:
        logging.error(f"Failed to get response from Groq API: {e}")
        return jsonify({"error": str(e)}), 500

def get_groq_response(conversation):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation,  # Send full chat history
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        return completion.choices[0].message.content  # Extract AI response
    
    except Exception as err:
        logging.error(f"An error occurred while communicating with Groq API: {err}")
        raise Exception("Failed to communicate with Groq AI.")

def store_chat_in_convex(user_message, bot_response):
    """Store user and bot messages in Convex database."""
    try:
        convex.insert("chats", {
            "user": user_message,
            "bot": bot_response
        })
        logging.info("Chat stored in Convex successfully.")
    except Exception as err:
        logging.error(f"Error storing chat in Convex: {err}")

if __name__ == "__main__":
    app.run(debug=True)
