# test_wordcollection.py

from tokenizer import WordCollection


def test_word_collection():
    collection = WordCollection()

    # add words
    words = ["run", "running", "runs",
             "played", "play", "plays",
             "happily", "happy"]

    for w in words:
        collection.add_word(w)

    run_idx = collection.encode("run")
    running_idx = collection.encode("running")
    runs_idx = collection.encode("runs")

    assert run_idx != running_idx
    assert run_idx != runs_idx
    assert running_idx != runs_idx

    assert collection.decode(run_idx) == "run"
    assert collection.decode(running_idx) == "running"
    assert collection.decode(runs_idx) == "runs"

    play_idx = collection.encode("play")
    played_idx = collection.encode("played")
    plays_idx = collection.encode("plays")

    assert play_idx != played_idx
    assert play_idx != plays_idx

    assert collection.decode(play_idx) == "play"
    assert collection.decode(played_idx) == "played"
    assert collection.decode(plays_idx) == "plays"

    happy_idx = collection.encode("happy")
    happily_idx = collection.encode("happily")

    assert happy_idx != happily_idx
    assert collection.decode(happy_idx) == "happy"
    assert collection.decode(happily_idx) == "happily"

    upper_idx = collection.encode("RUN")
    assert upper_idx == run_idx, "Lowercase normalization failed"

    expected_size = len(set([w.lower() for w in words]))
    assert collection.vocab_size() == expected_size

    print("All tests passed!")


if __name__ == "__main__":
    test_word_collection()