import logging
import os
import re
import string

import pandas as pd
from langid.langid import LanguageIdentifier, model

log = logging.getLogger(__name__)


def load_data_from_path(folder_path: str = None):
    """
    Load data from path

    Args:
        folder_path: str, the path reaching to data
    Returns:
        a pd.DataFrame with 2 fields: "sentence" and "label"
    """
    if not folder_path:
        log.error("Data folder path must be specific")
        return None
    examples = []
    for label in os.listdir(folder_path):
        # label include 2 values: 'pos' and 'neg'
        # create path to reach to label which contain file data
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
            sentences = " ".join(lines)
            if label == "pos":
                label = 1
            if label == "neg":
                label = 0
            data = {"sentence": sentences, "label": label}
            examples.append(data)
    log.info("Loaded data to Data Frame format")
    return pd.DataFrame(examples)


def identify_vn(df: pd.DataFrame = None):
    """
    this function is used to focus only on vietnamese data instead of
    other language. So, I will remove all of rows which don't contain
    vietnamese language from data frame

    Args:
        df: pd.DataFrame, previous dataframe which contains both vietnamese and
        non-vietnamese languages
    Returns:
        vi_df: pd.DataFrane, data frame only contains vietnamese language
    """
    if df is None:
        log.error("Input must be specific")
        return None
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    not_vn_idx = set()
    threshold = 0.9
    for idx, row in df.iterrows():
        score = identifier.classify(row["sentence"])
        if score[0] != "vi" or (score[0] == "vi" and score[1] <= threshold):
            not_vn_idx.add(idx)
    vi_df = df[~df.index.isin(not_vn_idx)]
    return vi_df


def preprocessing_text(text: str):
    """
    This function to used to remove special pattern from text,
    and replace with white space
    Args:
        text: str
    Returns:
        a clean text contain only vietnamese characters
    """
    # Defining the  url pattern
    url_pattern = re.compile(r"https?://\s+\www\.\s+")
    # Replace url with white space
    text = url_pattern.sub(r" ", text)

    # Defining the html pattern
    html_pattern = re.compile(r"<[^<>]+>")
    # Replace html pattern with white space
    text = html_pattern.sub(" ", text)

    # Defining punctuation and digits patern
    replace_chars = list(string.punctuation + string.digits)
    # Replace with white space
    for char in replace_chars:
        text = text.replace(char, " ")

    # Defining the emoji patern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F1F2-\U0001F1F4"  # Macau flag
        "\U0001F1E6-\U0001F1FF"  # flags
        "\U0001F600-\U0001F64F"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U0001F1F2"
        "\U0001F1F4"
        "\U0001F620"
        "\u200d"
        "\u2640-\u2642"
        "]+",
        flags=re.UNICODE,
    )
    # Replace with white space
    text = emoji_pattern.sub(r" ", text)

    # Remove duplicate white space
    text = " ".join(text.split())
    # return values with lower case
    return text.lower()


def preprocessing(df: pd.DataFrame):
    df["sentence"] = [
        preprocessing_text(row["sentence"]) for idx, row in df.iterrows()
    ]
    return df


if __name__ == "__main__":
    folderPath_ = "data/data_train/train"
    data_ = load_data_from_path(folderPath_)
    vi_data = identify_vn(data_)
    clean_vi_data = preprocessing(vi_data)
    print(clean_vi_data)
