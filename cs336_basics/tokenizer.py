import pickle
import regex as re
from typing import Iterable, Iterator


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(
            special_tokens or [],
            key=len,
            reverse=True,
        )

        self.reversed_vocab = {v: k for k, v in vocab.items()}
        self.merge_ranks = {merges[i]: i for i in range(len(merges))}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)

        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []

        if self.special_tokens:
            split_pattern = '|'.join(re.escape(t) for t in self.special_tokens)
            parts = re.split(f'({split_pattern})', text)
        else:
            parts = [text]

        pat = re.compile(PAT)
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.reversed_vocab[part.encode('utf-8')])
            else:
                for match in pat.finditer(part):
                    pretoken = tuple(bytes([b]) for b in match.group().encode('utf-8'))

                    while True:
                        pairs = zip(pretoken[:-1], pretoken[1:])

                        ranks = {pair: self.merge_ranks[pair] for pair in pairs if pair in self.merge_ranks}
                        if not ranks:
                            break

                        merge = min(ranks.items(), key=lambda kv:kv[1])[0]

                        i = 0
                        new_token = []
                        while i < len(pretoken):
                            if tuple(pretoken[i:i+2]) == merge:
                                new_token.append(pretoken[i] + pretoken[i+1])
                                i += 2
                            else:
                                new_token.append(pretoken[i])
                                i += 1

                        pretoken = tuple(new_token)

                    for token in pretoken:
                        ids.append(self.reversed_vocab[token])
        
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b''.join([self.vocab[id] for id in ids])

        return text.decode('utf-8', errors='replace')