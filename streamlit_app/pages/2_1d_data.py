import streamlit as st
import pandas as pd
import altair as alt

@st.cache_data()
def get_task_instruction(context, task):
    """add instructions specific to task to be performed"""
    if task == "annomaly detection":
        return context + """
    The model should detect annomalies in the data and output solely a list of booleans, 
    e.g. [True, False, False] indicating indices that are outliers.
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
st.write("to do exploration on 1d data (e.g. time series) could be used for interpretation, annomaly detection, interpolation, extrapolation,...")

# context to be added to system or user message (e.g. )
context = st.text_input(label="(optional) some context for interpretation", 
              value="You are presented equidistant time series data from water level measurements.", key="context")

# select task to be performed
task = st.selectbox("select task", ["annomaly detection", "interpolation", "extrapolation", "interpretation"], key="task")

# construct system message from context and task
message = get_task_instruction(context, task)
st.write(message)



st.write("----")
# create three columns with 1/6, 4/6, 1/6 width
col1, col2, col3 = st.columns([2,5,3])

with col1:
    st.write("**input**")
    initial_data = generate_initial_data(n=10)
    # experiment with data editor (can handle pandas, pyarrow, numpy, snowflake dataframe, list,....)
    input_data = st.experimental_data_editor(initial_data, num_rows='dynamic', height=300)
    
with col2:
    st.write("**completion**")
    st.text_area(label="completion", value="", key="completion", height=250, label_visibility='hidden')
    # INSERT COMPLETION HERE AND IN COLUMN 3,  INSERTED DUMMY FOR NOW
    list_output = input_data 

with col3:
    st.write("**tabular output**")
    st.experimental_data_editor(pd.DataFrame(list_output, columns=['output']), height=300)
  
  
  
st.write("----")    


compose_plot(input_data, list_output)
    