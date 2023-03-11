import streamlit as st

st.set_page_config(
    page_title="Hejsa",
    page_icon="👋",
)

st.write("# Welcome to OpenAI API Exploration")

st.sidebar.title("Navigation")

st.markdown(
    """
    The openAI API is a great tool for exploring the possibilities of Large language models. 
    This app is a playground for exploring the API and its capabilities. The app is built using Streamlit, a Python library for creating web apps. The app is hosted on Heroku, a cloud platform as a service supporting several programming languages. The app is open source and can be found on [GitHub](
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Use **classical chat** with access to parameters and system context
    - 1d data (timeseries) do exploration on  
        - interpretation
        - annomaly detection
        - interpolation
    - 2d data (images)
"""
)