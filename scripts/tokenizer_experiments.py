import random
import numpy as np
from cs336_basics.tokenizer import Tokenizer


random.seed(42)


def sample_documents(file_path, n=10, delimiter='<|endoftext|>'):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    docs = [doc for doc in text.split(delimiter) if doc.strip()]
    return random.sample(docs, n)


def load_tokenizer(vocab_path, merges_path):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens='<|endoftext|>'
    )

    return tokenizer

def compression_ratio(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        ids = tokenizer.encode(doc)
        total_bytes += len(doc.encode('utf-8'))
        total_tokens += len(ids)

    return total_bytes / total_tokens


def encode_dataset_to_uint16(tokenizer, input_path, output_path):
    token_ids = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for token_id in tokenizer.encode_iterable(f):
            token_ids.append(token_id)

    token_ids = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, token_ids)

    print(f'Saved {len(token_ids)} tokens to {output_path}')
    print(f'Max token id: {token_ids.max()}')


if __name__ == '__main__':
    tinystories_train = 'data//TinyStoriesV2-GPT4-train.txt'
    tinystories_vocab = 'data/output/tinystories_vocab.pkl'
    tinystories_merges = 'data/output/tinystories_merges.pkl'
    save_path = 'data/output/tinystories_encoded.npy'

    docs = sample_documents(tinystories_train)
    tokenizer = load_tokenizer(tinystories_vocab, tinystories_merges)
    ratio = compression_ratio(tokenizer, docs)

    print(f'compression ratio: {ratio}')

    encode_dataset_to_uint16(
        tokenizer=tokenizer,
        input_path=tinystories_train,
        output_path=save_path
    )




