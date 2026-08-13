from pathlib import Path
import streamlit as st
import app

st.set_page_config(page_title="Index Build Diagnostic")

st.title("Etymology index diagnostic")

data_path = Path(app.DEFAULT_DATA_FILE)

st.write("Data path:", str(data_path))
st.write("File exists:", data_path.exists())

if not data_path.exists():
    st.error("The JSONL data file is not available.")
    st.stop()

size_mb = data_path.stat().st_size / (1024 * 1024)
st.write(f"File size: {size_mb:.1f} MB")

st.success("JSONL file is available.")

if st.button("Build etymology index"):
    st.write("Starting index construction...")

    try:
        with st.spinner("Building etymology index..."):
            app.build_indices(str(data_path))

        st.success("Index construction completed successfully.")

        st.write("WORD_TO_ENTRY:", len(app.WORD_TO_ENTRY))
        st.write(
            "WORDS_BY_LANGUAGE:",
            sum(len(words) for words in app.WORDS_BY_LANGUAGE.values())
        )
        st.write("ANCESTOR_TO_ENTRIES:", len(app.ANCESTOR_TO_ENTRIES))
        st.write(
            "GLOBAL_ANCESTORS_BY_NORM:",
            len(app.GLOBAL_ANCESTORS_BY_NORM)
        )
        st.write(
            "ALL_ANCESTOR_RECORDS:",
            len(app.ALL_ANCESTOR_RECORDS)
        )

    except Exception as exc:
        st.error("Index construction raised an exception.")
        st.exception(exc)