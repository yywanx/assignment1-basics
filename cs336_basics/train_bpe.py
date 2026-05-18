import os
import multiprocessing
from xxlimited import new
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict


NUM_PROCESSES = 12
END_OF_FILE = b'<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_pretoken(inputs):
    text, special_tokens = inputs

    if special_tokens:
        split_pattern = '|'.join(re.escape(t) for t in special_tokens)
        parts = re.split(split_pattern, text)
    else:
        parts = [text]

    pretokens = []
    pat = re.compile(PAT)
    for part in parts:
        for match in pat.finditer(part):
            pretoken = tuple(bytes([b]) for b in match.group().encode('utf-8'))
            pretokens.append(pretoken)

    return Counter(pretokens)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. read file and chunk
    chunks = []
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, END_OF_FILE)
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            chunks.append([chunk, special_tokens])
    
    # 2. parallel processing
    pretoken_counter = Counter()
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        for counter in pool.imap_unordered(count_pretoken, chunks):
            pretoken_counter.update(counter)

    # 3. counting
    pair_counter = defaultdict(int)
    pair_to_pretoken = defaultdict(set)

    for pretoken, count in pretoken_counter.items():
        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_counter[pair] += count
            pair_to_pretoken[pair].add(pretoken)
    
    # 4. merge and update
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    merges = []
    
    while len(vocab) < vocab_size:
        most_freq_pair = max(pair_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merged = most_freq_pair[0] + most_freq_pair[1]
        connected_pretokens = pair_to_pretoken[most_freq_pair]

        merges.append(most_freq_pair)
        vocab[len(vocab)] = merged

        for pretoken in list(connected_pretokens):

            pretoken_count = pretoken_counter.pop(pretoken)
            for pair in zip(pretoken[:-1], pretoken[1:]):
                pair_to_pretoken[pair].discard(pretoken)

                pair_counter[pair] -= pretoken_count
                if pair_counter[pair] == 0:
                    del pair_counter[pair]

            new_token = []
            i = 0
            while i < len(pretoken):
                if tuple(pretoken[i:i+2]) == most_freq_pair:
                    new_token.append(merged)
                    i += 2
                else:
                    new_token.append(pretoken[i])
                    i += 1

            new_token = tuple(new_token)
            pretoken_counter[new_token] += pretoken_count

            for pair in zip(new_token[:-1], new_token[1:]):
                pair_counter[pair] += pretoken_count
                pair_to_pretoken[pair].add(new_token)

    return vocab, merges

            