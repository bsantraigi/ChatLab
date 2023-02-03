import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the maximum length of input text to be used for generating response
MAX_LENGTH = 256

def generate_response(input_text, context):
    input_text = f"{context} {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)
    response = model.generate(input_ids, max_length=MAX_LENGTH, 
                              num_return_sequences=1, top_p=0.9, top_k=50)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return response_text

st.title("ChatGPT")

# Keep track of the conversation history
conversation_history = []

# Get the user input
input_text = st.text_input("You:")

# Display the previous conversation history
for message in conversation_history:
    st.write(message)

# Generate and display the response
if input_text:
    context = " ".join(conversation_history)
    response = generate_response(input_text, context)
    st.write("Bot: " + response)
    conversation_history.append(f"You: {input_text}")
    conversation_history.append(f"Bot: {response}")

