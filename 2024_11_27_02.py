# %%
import streamlit as st

# %%
# session
if 'counter' not in st.settion_state:
    st.session_state.counter = 0

# counter = 0

if st.button('increase'):
    st.session_state.counter += 1

st.write(f'counter: {st.session_state.counter}')

# %%
