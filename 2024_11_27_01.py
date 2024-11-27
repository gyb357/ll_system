# %%
import streamlit as st

# %%
with st.sidebar:
    st.title('sidebar')
    _radio = st.radio(
        'choose method',
        ['method1', 'method2']
    )
    msg = st.text_input(
        label='enter message',
        placeholder='input any message'
    )
if st.button('submit'):
    st.write(f'radio: {_radio}')
    st.write(f'message: {msg}')
    