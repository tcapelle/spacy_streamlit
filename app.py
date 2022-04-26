"""
Example using the components provided by spacy-streamlit in an existing app.
Prerequisites:
python -m spacy download en_core_web_sm
"""
import spacy_streamlit, wandb
import streamlit as st

DEFAULT_TEXT = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."""


st.title("NER using spaCy")

# key = st.text_input("Paste your wandb API key here https://wandb.ai/authorize")
# wandb.login()

wandb.login(anonymous="must")

ENTITY = "capecape"
PROJECT = "st30"

api = wandb.Api()
artifacts_type = api.artifact_type("model", f'{ENTITY}/{PROJECT}')

def list_project_models(artifacts_type):
    models = []
    for collection in artifacts_type.collections():
        for artifact in collection.versions():
            models.append(artifact.name)
    return models

models_names = list_project_models(artifacts_type)
model_name = st.selectbox("Select your spaCy model (logged as wandb.Artifact)", models_names)

# download the model from wandb
model = api.artifact(f'{ENTITY}/{PROJECT}/{model_name}', type='model')
model = model.download()

text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)
doc = spacy_streamlit.process_text(model, text)

ner_labels = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", 
              "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", 
              "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]

spacy_streamlit.visualize_ner(
    doc,
    labels=ner_labels,
    show_table=False,
    title="Persons, dates and locations",
)
st.text(f"Analyzed using spaCy model {model}")
