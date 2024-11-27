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

# %%
radio = st.radio('Radio', ['A', 'B', 'C'])

# %%
check1 = st.checkbox('check1')
check2 = st.checkbox('check2')
st.divider()

# %%
age = st.slider('Age', 0, 130, 25)
st.write(f'i am {age} years old')

# %%
if st.button('ok'):
    print(f'checkbox1: {check1}')
    print(f'checkbox2: {check2}')
    print(f'radio: {radio}')
    print(f'age: {age}')

# %%
