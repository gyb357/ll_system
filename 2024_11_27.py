# %%
import streamlit as st

# %%
test_val = 'hello'
test_val += 'python'

# %%
st.title('UI Test')
st.text('it is plain text')

# %%
if st.button('click me'):
    st.text('button clicked')

# %%
agree = st.checkbox('I agree')

# %%
if agree:
    st.text('you agreed')
else:
    st.text('you did not agree')
