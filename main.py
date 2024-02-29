import streamlit as st
import textwrap
import langchain_helper as lch


c = st.container()
c.title('YT Acharya 🧑🏻‍🏫')

def display():
            if query and youtube_url:
                db = lch.create_vector_db_from_url(youtube_url)
                if db == "No transcript":
                    c.subheader('Sorry the url you provided dosen\'t have a transcript. Trying to get context with the help of audio , please be patient  🙇🏻')
                    db = lch.get_audio(youtube_url)
                    response  = lch.get_response_for_query(db,query)
                    c.subheader('Answer')
                    c.text(textwrap.fill(response,width=80))
                else:
                    response  = lch.get_response_for_query(db,query)
                    c.subheader('Answer')
                    c.text(textwrap.fill(response,width=80))



youtube_url = st.sidebar.text_area(
            label="What is the Youtube Video URL?",
            max_chars=50
        )
query = st.sidebar.text_area(
            label = "What do you want to ask?",
            max_chars=50,
            key="query"
            )
button = st.sidebar.button(label='Submit',on_click=display)

