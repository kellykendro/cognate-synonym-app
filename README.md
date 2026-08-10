# Cognate Synonym Evaluation — Browser Streamlit App

This project is a browser-hosted Streamlit conversion of the original Flask/Jinja application.

## Project files

- `app.py` — Streamlit application and analysis backend
- `etymology_data_unique_cleaned.jsonl` — etymology dataset
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server configuration
- `.gitattributes` — tracks the large JSONL file with Git LFS
- `.gitignore` — excludes local environment and secret files
- `reference/index.html` — preserved copy of the original Flask/Jinja interface for visual reference; Streamlit does not execute it

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal, normally `http://localhost:8501`.

## Deploy as a browser app with Streamlit Community Cloud

The JSONL file is about 127 MB, so it must be pushed to GitHub with Git LFS rather than normal Git tracking.

### 1. Create an empty GitHub repository

Create a new repository on GitHub. Do not add a README or other files there if you plan to push this folder directly.

### 2. Push this folder with Git LFS

From a terminal opened inside this project folder:

```bash
git lfs install
git init
git lfs track "*.jsonl"
git add .
git commit -m "Initial Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

If the JSONL was accidentally committed with ordinary Git before Git LFS was configured, remove it from ordinary Git history or migrate it to LFS before pushing.

### 3. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app**.
3. Select your GitHub repository.
4. Select branch `main`.
5. Set the entrypoint to `app.py`.
6. In **Advanced settings**, use Python 3.12.
7. Click **Deploy**.

Streamlit will install the dependencies from `requirements.txt`, fetch the Git LFS dataset, and give the app a public `streamlit.app` browser URL.

## Interface fidelity

The Streamlit interface is deliberately styled to closely reproduce the original `index.html`: the bordered input form is back in the main page, the original Arial/white visual treatment is retained, results use the same five-column table, direct-cognate alerts use the same green treatment, and the semantic/orthographic charts use the original vertical histogram layout. Streamlit still owns the actual input widgets, so minor browser-level differences in sliders, selects, and file upload controls are expected.

## App behavior

- Spanish remains the enabled L2 language, matching the original Python configuration.
- A single manually entered word can use a supplied translation.
- Batch `.txt` files use one English word per line and automatic translation.
- Semantic, orthographic, and frequency thresholds are interactive.
- Histograms and candidate tables update without rerunning the linguistic analysis.
- CSV export reflects the currently selected thresholds.
- The large etymology index is cached with Streamlit resource caching to avoid rebuilding it on ordinary widget reruns.
- `sentence-transformers`, `wordfreq`, and `deep-translator` preserve the original behavior, with graceful fallbacks where the backend supports them.

## Data path

By default the app expects:

```text
etymology_data_unique_cleaned.jsonl
```

beside `app.py`. You can override it with the `ETYMOLOGY_DATA_FILE` environment variable when hosting somewhere other than Community Cloud.

## Notes for cloud resources

The first start is the most expensive because the JSONL index must be constructed and the sentence-transformer model may need to load. Community Cloud has finite CPU and memory, so if the app exceeds its resource allocation, the next optimization should be to precompute a compact index or move the dataset/index to external storage rather than rebuilding all structures from JSONL on startup.
