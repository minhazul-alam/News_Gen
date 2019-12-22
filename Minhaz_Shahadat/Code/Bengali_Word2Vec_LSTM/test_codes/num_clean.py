import re

vocab_path = 'f:/vocab.txt'
output_path = 'f:/output.txt'


def open_vocab_file_as_list():
    print("Reading the vocab file...")
    with open(vocab_path, 'r', encoding="utf-8") as file:
        vocab = file.readlines()

    vocab = [word.strip() for word in vocab]
    return vocab


def clean_number_from_vocab(vocab_list):
    print("Cleaning the vocab list...")
    vocab = list(map(lambda x: x if len(re.findall(r'\d+', x)) == 0 else '<number>', vocab_list))
    return vocab


def save_cleaned_vocab_list(vocab_list):
    print("Saving the cleaned vocab list...")
    with open(output_path, 'w', encoding="utf-8") as f:
        for word in vocab_list:
            f.write("%s\n" % word)


if __name__ == "__main__":
    vocab_list = open_vocab_file_as_list()
    cleaned_list = clean_number_from_vocab(vocab_list)
    save_cleaned_vocab_list(cleaned_list)