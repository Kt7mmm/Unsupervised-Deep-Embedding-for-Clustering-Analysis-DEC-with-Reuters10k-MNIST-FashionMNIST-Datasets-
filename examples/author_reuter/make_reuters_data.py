import os
import numpy as np

def make_reuters_data(data_dir):
    """
    Load Reuters dataset.

    Args:
    data_dir: Directory containing the Reuters dataset files.

    Returns:
    X: List of documents represented as lists of words.
    y: List of document labels.
    """
    X = []  # List to store documents
    y = []  # List to store labels

    # Loop through the data files in the directory
    for filename in os.listdir(data_dir):
        # Check if the file is a data file
        if filename.startswith("lyrl2004_tokens_train") or filename.startswith("lyrl2004_tokens_test"):
            with open(os.path.join(data_dir, filename), "r", encoding="latin-1") as file:
                lines = file.readlines()
                for line in lines:
                    # Each line represents a document
                    # Split the line by whitespace to get words
                    words = line.strip().split()
                    # The first word is the document label
                    label = int(words[0])
                    # The remaining words are the document content
                    content = words[1:]
                    X.append(content)
                    y.append(label)

    return X, y

# Example usage
data_dir = r"examples\author_reuter"
X, y = make_reuters_data(data_dir)
print("Number of documents:", len(X))
print("Number of labels:", len(set(y)))
