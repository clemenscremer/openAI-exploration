import streamlit as st
import os
import openai
import tiktoken
import json


# switch layout to wide
st.set_page_config(layout="wide")	

openai.api_key = os.getenv("OPENAI_API_KEY")

# choose model and setup encoding (for token count)
model = "gpt-3.5-turbo"
encoding_name = "cl100k_base" # for chatGPT models (e.g. gpt-3.5-turbo)

@st.cache(allow_output_mutation=True)
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# initialize
st.header("Chat and Speech Assistant")

st.subheader("Initialize behavior and context")
st.write("If behavior and context is changed, the chatbot will be reinitialized.")
behavior = st.text_input("", "You are a truthful scientist communicating to a general audience", 
                         key="behavior")
@st.cache(allow_output_mutation=True)
def initialize_chatbot(behavior):
    messages = [
        {"role": "system", 
        "content": behavior}, # could also be sarcastic troll
    ]
    return messages
messages = initialize_chatbot(behavior)

# alternatively initialize from logfile (if possible)


# chat
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prompt")
    prompt = st.text_area("", height=300, key="prompt")

    num_tokens = num_tokens_from_string(prompt, encoding_name)
    st.write(f"Number of tokens: {num_tokens}")


@st.cache(allow_output_mutation=True)
def get_assistant_response(prompt, model):
    messages.append(
        {"role": "user", "content": prompt},
        )

    # define assistant model and process output
    chat = openai.ChatCompletion.create(
        model=model, # "gpt-3.5-turbo" same as in current chatgpt app 
        messages=messages,
        max_tokens=1024, # limit max. token spending
        temperature=1,   # randomness of assistant. default = 1 
        )
    
    # extract first response [0]
    reply = chat.choices[0].message.content
    # print assistant response
    messages.append(
        {"role": "assistant", "content": reply}
        )
    return reply

reply = get_assistant_response(prompt, model)
with col2:
    st.subheader("Completion")
    completion = st.text_area(f"", height=300, key="completion", value=reply)
    num_tokens = num_tokens_from_string(completion, encoding_name)
    st.write(f"Number of tokens: {num_tokens}")


st.write(messages)


# save messages to logfile (maybe enable reload later)
@st.cache
def save_logfile(messages):
    with open("logfile.json", "w") as f:
        json.dump(messages, f)
save_logfile(messages)
