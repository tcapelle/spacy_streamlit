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
_models = ["en_core_web_sm", "en_core_web_md"]

models = []
for model in _models:
    artifact = wandb.Api().artifact(f'capecape/st30/{model}:v0', type='model')
    models.append(artifact.download())

model = st.selectbox("Select your spaCy model artifact", models)

artifact = wandb.Api().artifact('capecape/st30/spacy_model:v0', type='model')
model = artifact.download()


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
