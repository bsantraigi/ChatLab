import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer

import os
import openai

# Option to select a model from the list of available models
st.sidebar.title("Select a model")
model_name = st.sidebar.selectbox("Model", ("Select a model", "gpt2", "flan-t5-base", "openai/chatgpt"))

if model_name == "Select a model":
    st.stop()

# Show saved sessions
st.sidebar.title("Saved sessions")
# Retrieve the saved sessions from sqlite database
import sqlite3
conn = sqlite3.connect('sessions.db')
c = conn.cursor()
c.execute("SELECT * FROM sessions")
sessions = c.fetchall()
conn.close()
# Show the saved sessions in a dropdown
session_name = st.sidebar.selectbox("Session", ("Select a session",) + tuple([session[0] for session in sessions]))
if session_name != "Select a session":
    # Load the conversation history from the selected session
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE name=?", (session_name,))
    session = c.fetchone()
    conn.close()
    print(session)

if model_name != "openai/chatgpt":
    # Convert model name to huggingface model name
    converter = {
        "gpt2": "gpt2",
        "flan-t5-base": "google/flan-t5-base"
    }
    model_name = converter[model_name]

    if model_name.startswith("gpt2"):
        # Load the GPT-2 model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name.startswith("google"):
        # Load the T5 model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    # # Load the GPT-2 model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    # Set the maximum length of input text to be used for generating response
    MAX_LENGTH = 256

    def generate_response(prompt):
        # input_text = f"{context}\n{input_text}"
        input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
        response = model.generate(input_ids, max_length=MAX_LENGTH, 
                                num_return_sequences=1, top_p=0.9, top_k=50)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        return response_text


else:
    # will call openai api directly
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt="",
    #     temperature=0.7,
    #     max_tokens=256,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )

st.title("ChatGPT")

# Keep track of the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.session_state

# Display the previous conversation history
for message in st.session_state.conversation_history:
    history_text = st.write(message)

# Get the user input
input_text = st.text_area("You:")

# Generate and display the response
if input_text:
    # for message in st.session_state.conversation_history:
    #     st.write(message)
    input_text = f"User: {input_text}"
    prompt = "\n".join(st.session_state.conversation_history + [input_text])
    response = generate_response(prompt + "\nAgent:")
    st.session_state.conversation_history.append(input_text)
    st.session_state.conversation_history.append(f"Agent: {response}")
    st.write(f"{model_name}: " + response)

# Save the conversation history
if st.button("Save session"):
    # Use the model to create a name for the session
    name_prompt = """Suggest a name for this session:
    """
    name_prompt += "\n".join(st.session_state.conversation_history)
    name_prompt += "\nName:"
    session_name = generate_response(name_prompt)
    # Upto 5 words for session name
    session_name = " ".join(session_name.split()[:5])

    # If session name is empty, use "Session {row_id}"
    if not session_name.strip():
        conn = sqlite3.connect('sessions.db')
        c = conn.cursor()
        c.execute("SELECT * FROM sessions")
        sessions = c.fetchall()
        conn.close()
        session_name = f"Session {len(sessions) + 1}"

    # Save the conversation history to sqlite database
    conn = sqlite3.connect('sessions.db')
    c = conn.cursor()
    # id (autoincrement), name, conversation_history, created_at
    c.execute("INSERT INTO sessions VALUES (NULL, ?, ?, datetime('now'))", (session_name, "\n".join(st.session_state.conversation_history)))
    conn.commit()
