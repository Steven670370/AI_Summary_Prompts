# test_wordcollection.py

from tokenizer import WordCollection

def test_word_collection():
    collection = WordCollection()

    # add words
    words = ["run", "running", "runs", "played", "play", "plays", "happily", "happy"]
    for w in words:
        collection.add_word(w)

    # check if words are nomarlized
    run_idx = collection.encode("run")
    running_idx = collection.encode("running")
    runs_idx = collection.encode("runs")

    assert run_idx == running_idx == runs_idx, "run variants should have the same encoding"

    play_idx = collection.encode("play")
    played_idx = collection.encode("played")
    plays_idx = collection.encode("plays")

    assert play_idx == played_idx == plays_idx, "play variants should have the same encoding"

    happy_idx = collection.encode("happy")
    happily_idx = collection.encode("happily")

    assert happy_idx == happily_idx, "happy variants should have the same encoding"

    # return base
    assert collection.decode(run_idx) == "run", "decode(run) should return 'run'"
    assert collection.decode(play_idx) == "play", "decode(play) should return 'play'"
    assert collection.decode(happy_idx) == "happy", "decode(happy) should return 'happy'"

    # check the size of table
    vocab_size = collection.vocab_size = len(collection.words)
    expected_size = 3  # run/play/happy
    assert vocab_size == expected_size, f"vocab size should be {expected_size}"

    print("All tests passed!")

if __name__ == "__main__":
    test_word_collection()