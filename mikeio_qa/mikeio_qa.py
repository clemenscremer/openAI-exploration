import streamlit as st
import pandas as pd
import numpy as np
import openai

from openai.embeddings_utils import distances_from_embeddings

#st.set_page_config(layout="wide")



# -------------------------------------- data --------------------------------------
# read in data from embeddings (previously generated with openai API)
df = pd.read_csv('../notebooks/processed/embeddings.csv')
# convert embeddings to 1d numpy array, makes it easier to work with later
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)



# -------------------------------------- model --------------------------------------
# input sidebar
engine = 'text-embedding-ada-002' # SET engine for embeddings
completion_model = st.sidebar.selectbox("completion model", 
                                        ("text-davinci-003", "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001")
                                        ) # needs to be completion (single-turn) model (ada, babbage, curie or davinci) and not chat model see https://platform.openai.com/docs/models/overview on completion models
max_len = st.sidebar.number_input(label="max. content length", value=1800) # max length of context
max_tokens = st.sidebar.number_input(label="max. answer tokens", value=1000, 
                                     help="""Tip: Use max_tokens > 256. The model is better at inserting longer completions. 
                                     With too small max_tokens, the model may be cut off before it's able to connect to the suffix""") # maximum number of tokens to answer the question
st.sidebar.markdown("-----")
temperature = st.sidebar.number_input(value=0.0, min_value=0.0, max_value=2.0, step=0.1, label="temperature", 
                                      help="""What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, 
                                      while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.""") # temperature for completion model
top_p = st.sidebar.number_input(value=1.0, min_value=0.1, max_value=1.0, step=0.1, label="top p", 
                                help="""An alternative to sampling with temperature, called nucleus sampling, where the model considers the
                                results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
                                We generally recommend altering this or temperature but not both.""")
frequency_penalty = st.sidebar.number_input(value=0.0, min_value=-2.0, max_value=2.0, step=0.1, label="frequency penalty", 
                                            help="""Number between -2.0 and 2.0. Positive values penalize new tokens based on 
                                            their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.""")
presentence_penalty = st.sidebar.number_input(value=0.0, min_value=-2.0, max_value=2.0, step=0.1, label="presentence_penalty",
                                              help="""Number between -2.0 and 2.0. Positive values penalize new tokens based
                                            on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.""")



# check if 8191 tokens are exceeded by question + context + answer
if max_len + max_tokens > 8191:
    st.write("max_len + max_tokens exceeds 8191 tokens")


def create_context(question, df, max_len=max_len, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine=engine)['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

# for completions and parameters see https://platform.openai.com/docs/api-reference/completions/create?lang=python
def answer_question(
    df,
    model=completion_model,
    question="Why should i use mikeio?",
    max_len=max_len,
    size="ada",
    debug=False,
    max_tokens=max_tokens,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"""Answer the question based on the context below, 
            and if the question can't be answered based on the context, 
            say \"I don't know\". 
            Answer in nicely formatted markdown.
            Take note of the sources and include them in the answer 
            in the format: "SOURCES: source1 source2", 
            use "SOURCES" in capital letters regardless of the number of sources.
            Seperate sources with a space from body of answer.
            In case you provide example code snippets in the answer, separate them with newline.
            \n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:""",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presentence_penalty,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    
    # ------------------------------------ app / gui -------------------------------------




# ------------------------------------ main app / gui -------------------------------------
hd1, hd2 = st.columns(2)
# insert image
st.image("https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png", width=300)	    

# question text box
st.subheader("Ask me anything about MIKEIO")
st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-image: linear-gradient(180deg, rgb(255, 255, 255) 0%, rgb(255, 255, 255) 80%, rgb(255, 255, 255)  100%);
    }</style>
    """,unsafe_allow_html=True)
question = st.text_area("Why should i use MIKEIO?", 
                        height=150, 
                        key="question", 
                        label_visibility="hidden", 
                        disabled=False)

#st.subheader("Answer")
# get reply only if prompt is not empty
if question:
    answer = answer_question(df, question=question)#, debug=True)
else:    
    answer = ""

if answer != "":    
    st.subheader("Here is my answer:")
    st.markdown(answer)
    
    


# ------------------------------------ misc -------------------------------------
st.markdown("----- \n **misc**")

with st.expander("**show log**"):
    st.write(f"""Question:{question} \n\nAnswer:{answer}""")
    
with st.expander("**more information**"):
    st.write("""we provide single shot answers. 
                Chat capabilities (iterative Q&A) are not supported yet. Sources can still be untrue or misleading.\
                To create the embeddings we use the openai api and the embeddings model \"text-embedding-ada-002\".\
                The used model is currently (April 2023) ([outperforming all prior embedding models](https://openai.com/blog/new-and-improved-embedding-model)).
                """)