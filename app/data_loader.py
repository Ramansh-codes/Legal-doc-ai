from datasets import load_dataset

def load_legal_dataset():
    ds = load_dataset("joelniklaus/legal_case_document_summarization")
    return ds["train"]