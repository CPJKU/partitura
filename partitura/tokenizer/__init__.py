from .remi import REMITokenizer

SUPPORTED_ENCODINGS = ["REMI"]

def tokenizer(encoding_scheme="REMI", config=None):
    if encoding_scheme == "REMI":
        return REMITokenizer(config)
    else:
        raise ValueError(
            f"Unknown encoding scheme: '{encoding_scheme}'. "
            f"Currently supported schemes are: {SUPPORTED_ENCODINGS}"
        )