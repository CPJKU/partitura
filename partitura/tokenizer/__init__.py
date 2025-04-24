from .remi import REMITokenizer

def tokenizer(encoding_scheme="REMI", config=None):
    if encoding_scheme == "REMI":
        return REMITokenizer(config)
    else:
        raise ValueError(f"Unknown encoding scheme: {encoding_scheme}")