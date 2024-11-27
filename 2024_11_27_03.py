# %%
import streamlit as st
import os

# %%
base_path = '/streamlit/'

st.title('upload file')

if not os.path.exists(base_path):
    os.makedirs(base_path)

if 'files' not in st.session_state:
    st.session_state.files = []

if st.button('save_file'):
    for upload_file in st.session_state.files:
        file_content = upload_file.read()
        file_path = os.path.join(base_path, upload_file.name)
        with open(file_path, 'wb') as f:
            f.write(file_content)

    if st.button('back'):
        pass
else:
    upload_file = st.file_uploader('choose upload file', accept_multiple_files=True)

    if upload_file is not None:
        st.session_state.files = upload_file


# %%
