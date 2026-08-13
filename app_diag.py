import streamlit as st

st.set_page_config(page_title="Cognate App Diagnostic")

st.title("Cognate app diagnostic")

st.write("Testing import of the full application...")

try:
    import app
    st.success("app.py imported successfully.")
    st.write("No etymology index has been built yet.")
except Exception as exc:
    st.error("app.py failed during import.")
    st.exception(exc)