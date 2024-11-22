# %%
# %pip install streamlit

# %%
import streamlit as st

# %%
st.title('My first app')

# %%
st.text('Hello, world!')

# %%
st.markdown('Streamlit is **_really_ cool**.')

# %%
st.latex('a+ar + a r^2 + a r^3')

# %%
prompt = st.chat_input('Say something')
# editor_text = st.text_area('Editor', 'Type here...', key='editor1', value='default text')
editor_text = st.text_area('Editor', 'Type here...', key='editor1')

if prompt:
    print(prompt)
    st.text(f'You said: {prompt}')

# %%
