from nltk.tokenize import sent_tokenize

def read_tokenize_file(path):
    with open(path, "r") as f:
        text = f.read().replace("\n", " ")
        return sent_tokenize(text)