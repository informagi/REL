from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def load_bert_ner(path_or_url):
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_or_url)
        model = AutoModelForTokenClassification.from_pretrained(path_or_url)
        return pipeline("ner", model=model, tokenizer=tokenizer)
    except Exception:
        pass
    return
