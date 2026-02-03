from aigt.windowing import chunk_text_adaptive_windows


def test_chunking_nonempty():
    class DummyTok:
        def __call__(self, text, add_special_tokens, return_offsets_mapping, return_attention_mask, truncation):
            # naive "tokenizer": each word is a token, offsets cover the word spans
            words = text.split()
            ids = list(range(len(words)))
            offsets = []
            idx = 0
            for w in words:
                start = text.find(w, idx)
                end = start + len(w)
                offsets.append((start, end))
                idx = end
            return {"input_ids": ids, "offset_mapping": offsets}

    tok = DummyTok()
    text = "Hello world. This is a test. Another sentence here."
    wins = chunk_text_adaptive_windows(tok, text, token_length=3, keep_segment_text=True, min_tokens=1)
    assert len(wins) >= 1
    assert wins[0]["token_length"] <= 3
