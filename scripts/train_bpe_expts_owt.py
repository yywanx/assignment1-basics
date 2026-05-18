import time
import pickle
import cProfile
import pstats
from cs336_basics.train_bpe import train_bpe


def serialize(vocab, merges, save_dir):
    with open(save_dir + '/' + 'owt_vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open(save_dir + '/' + 'owt_merges.pkl', 'wb') as f:
        pickle.dump(merges, f)

    print(f'Serialized vocab and merges to {save_dir}')


if __name__ == '__main__':
    file_path = 'data/owt_train.txt'
    vocab_size = 32000
    special_tokens = ['<|endoftext|>']
    save_dir = 'data/output'

    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    end = time.perf_counter()
    print(f'totoal execution time: {end - start: .4f} seconds')

    serialize(vocab, merges, save_dir)

    longest_token = max(vocab.items(), key=lambda kv: len(kv[1]))[1]
    print(f'longest token is: {longest_token}')

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('time')
    stats.print_stats(30)