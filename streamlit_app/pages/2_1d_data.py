import streamlit as st
import pandas as pd
import altair as alt
import openai
import tiktoken
import os
# setup choose model and setup encoding (for token count)
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"
encoding_name = "cl100k_base" # for chatGPT models (e.g. gpt-3.5-turbo)

# functions
@st.cache_data()
def get_task_instruction(context, task):
    """add instructions specific to task to be performed"""
    if task == "annomaly detection":
        return context + """
    Identify annomalies in input sequence. 
    Limit your response strictly to a list of booleans e.g. [True, False, False] indicating found annomalies matching the inputs indices.
    """
    elif task == "interpolation":
        return "The model should interpolate missing values in the data."
    elif task == "extrapolation":
        return "The model should extrapolate the data by 10 values."
    elif task == "interpretation":
        return "The model should interpret the data."

def generate_initial_data(n=10):
        # generate list of fibonacci numbers as dummy data
    initial_data = [0, 1]
    for i in range(n-2):
        initial_data.append(initial_data[i] + initial_data[i+1])
    initial_data = initial_data + initial_data[::-1] # DUMMY DATA. CAN BE EXCHANGED LATER WITH REAL DATA
    return initial_data

@st.cache_data()
def initialize_chatbot(behavior):
    messages = [
        {"role": "system", 
        "content": behavior}, # could also be sarcastic troll
    ]
    return messages

@st.cache_data()
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@st.cache_data()
def get_assistant_response(prompt, model):
    messages.append(
        {"role": "user", "content": prompt},
        )

    chat = openai.ChatCompletion.create(
        model=model, 
        messages=messages,
        max_tokens=1024, 
        temperature=1,   # randomness of assistant. default = 1 
        )
    
    # extract first response [0]
    reply = chat.choices[0].message.content
    # print assistant response
    messages.append(
        {"role": "assistant", "content": reply}
        )
    return reply

@st.cache_data()
def compose_plot(input_data, list_output):
    # plot input and output
    df_in = pd.DataFrame(input_data, columns=['input'])
    df_out = pd.DataFrame(list_output, columns=['output'])
    df = pd.concat([df_in, df_out], axis=1, ignore_index=True)
    df.columns = ['input', 'output']
    df.index.name = 'index'	
    # df to wide form for altair
    data = df.reset_index().melt('index')


    # plot data
    c = alt.Chart(data).mark_point().encode(
        x='index',
        y='value',
        color='variable'
    )
    st.altair_chart(c, use_container_width=True)
    
    

st.title("1d data")
st.write("""
         to do exploration on 1d data (e.g. time series) could be used for interpretation, 
         annomaly detection, interpolation, extrapolation. Can also be helpful in detecting inconsistencies in inputs.""")



# initialize assistant: construct system message from context and task
context = st.text_input(label="(optional) some context for interpretation", 
              value="", key="context")
task = st.selectbox("select task", ["annomaly detection", "interpolation", "extrapolation", "interpretation"], key="task")
behavior = get_task_instruction(context, task)
messages = initialize_chatbot(behavior)
#st.write(behavior) # JUST TO CHECK, ERASE LATER



st.write("----")
# create three columns with 1/6, 4/6, 1/6 width
col1, col2, col3 = st.columns([2,5,3])

with col1:
    st.write("**input**")
    initial_data = generate_initial_data(n=10)
    # experiment with data editor (can handle pandas, pyarrow, numpy, snowflake dataframe, list,....)
    input_data = st.experimental_data_editor(initial_data, num_rows='dynamic', height=300)
    num_tokens = num_tokens_from_string(str(input_data), encoding_name)
    st.write(f"Number of tokens: {num_tokens}")  
with col2:
    #st.write("**input**")
    st.text_area(label="input", value=input_data, key="input", height=125)

    #st.write("**completion**")
    if input_data:
        reply = get_assistant_response(str(input_data), model)
        #reply = str(initial_data)
    else:    
        reply = ""
    st.text_area(label="completion", value=reply, key="completion", height=125)
    list_output = pd.DataFrame(reply[1:-1].split(","), columns=['output']) 
    #list_output = list_output.astype(bool)
    
with col3:
    st.write("**tabular output**")
    st.experimental_data_editor(list_output, height=300)
    num_tokens_out = num_tokens_from_string(reply, encoding_name)
    st.write(f"Number of tokens: {num_tokens_out}")
  
  
  
st.write("----")    

compose_plot(input_data, list_output)

# convert list output dataframe to dtype bool
list_output['output']  = list_output['output'].map({'True': True, 'False': False})
st.write(list_output.count())	    


# add expandable
with st.expander("Show log and messages"):	
    st.write(messages)