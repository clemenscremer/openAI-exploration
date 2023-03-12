import streamlit as st
import pandas as pd
import altair as alt
import openai
import tiktoken
import os
import regex as re
import numpy as np # NOT NEEDED ANYMORE
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
    initial_data = initial_data # DUMMY DATA. CAN BE EXCHANGED LATER WITH REAL DATA
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
def extract_list_from_reply(reply):
    # isolate list in reply
    reply_extract = re.findall(r"\[([^\]]*)\]", reply)[0].split(',') 
    # loop through list and eliminate whitespaces
    for i in range(len(reply_extract)):	
        reply_extract[i] = reply_extract[i].strip()	
    df_reply = pd.DataFrame(reply_extract, columns = ['output'])
    df_reply['output']  = df_reply['output'].map({'True': True, 'False': False})
    df_reply.index.name = 'index'	
    return df_reply

@st.cache_data()
def compose_plot(input_data, df_reply):
    # do some formatting (wide form for altair)
    df_in = pd.DataFrame(input_data, columns=['input'])
    df_in.index.name = 'index'
    data = df_in.reset_index().melt('index')
    

    # plot data
    c = alt.Chart(data).mark_point().encode(
        x='index',
        y='value',
        color='variable'
    )
    
    # handle and plot annomalies 
    annomalies = df_reply[df_reply['output'] == True].index.tolist()	    
    data_annomaly = data[data['index'].isin(annomalies)]	
    
    ol = alt.Chart(data_annomaly).mark_point(
        color='red', size=300, tooltip="annomaly").encode(
        x='index',
        y='value'
    )
        
    st.altair_chart(c + ol, use_container_width=True)

    
    

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
    # use regex to find list in reply
    list_output = pd.DataFrame((re.findall(r"\[([^\]]*)\]", reply)[0]).strip().split(','), columns=['output'])
    #st.write(list_output.astype(bool))    
    
    #list_output = pd.DataFrame(reply[1:-1].split(","), columns=['output']) 
    #list_output = list_output.astype(bool)
    
with col3:
    st.write("**tabular output**")
    st.experimental_data_editor(list_output, height=300)
    num_tokens_out = num_tokens_from_string(reply, encoding_name)
    st.write(f"Number of tokens: {num_tokens_out}")
  

  
st.write("----")    


df_reply = extract_list_from_reply(reply)

compose_plot(input_data, df_reply)


# add expandable
with st.expander("Show log and messages"):	
    st.write(messages)