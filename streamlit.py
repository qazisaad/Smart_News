import streamlit as st

options = st.multiselect(
    'Select your favorite categories',
    ['Green', 'Yellow', 'Red', 'Blue'],
    [])

st.write('You selected:', options)