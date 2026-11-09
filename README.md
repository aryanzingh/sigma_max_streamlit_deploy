# Sigma Max Predictor

Streamlit app that predicts maximum stress (Sigma Max, MPa) for a perforated plate.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (the model file is 67 MB, under GitHub's 100 MB limit).
2. Go to https://share.streamlit.io -> Create app -> pick the repo, branch, and `app.py`.
3. Under Advanced settings, choose Python 3.12 (scikit-learn 1.5.1 has no Python 3.13 wheels).
4. Deploy.

Do not change the scikit-learn version: the model was trained with 1.5.1 and must be loaded with the same version.
