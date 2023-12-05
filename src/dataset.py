from torchtext.data.utils import get_tokenizer

from utils.preprocessing import identify_vn, load_data_from_path, preprocessing
from utils.representation import build_vocabulary


def prepare_data(data_path):
    df = load_data_from_path(data_path)
    vi_df = identify_vn(df)
    clean_vi_df = preprocessing(vi_df)
    return clean_vi_df


def prepare_dataset(df, vocabulary, tokenizer):
    for index, row in df.iterrows():
        sentence = row["sentence"]
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row["label"]
        yield encoded_sentence, label


if __name__ == "__main__":
    folder_paths = {
        "train": "data/data_train/train",
        "valid": "data/data_train/test",
        "test": "data/data_test/test",
    }

    train_df = prepare_data(folder_paths["train"])
    valid_df = prepare_data(folder_paths["valid"])
    test_df = prepare_data(folder_paths["test"])

    tokenizer = get_tokenizer("basic_english")
    vocabulary = build_vocabulary(train_df["sentence"], tokenizer)

    train_dataset = prepare_dataset(train_df, vocabulary, tokenizer)

    print(train_dataset)
