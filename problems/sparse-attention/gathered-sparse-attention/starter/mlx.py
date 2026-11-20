from typing import Any

import mlx.core as mx


def sparse_attend(q: mx.array, store: Any, idx: mx.array) -> mx.array:
    """Attention of T queries, each over its own selected tokens only.

    q: float32, shape (T, H, d). One query per token and head.
    store: the keys and values of S tokens, (S, H, d) each, behind a counter:
        store.gather(rows, valid=None)  rows: integers of any shape -> (k_rows, v_rows),
                                        each of shape (*rows.shape, H, d); every entry
                                        read counts one row. Where the boolean array
                                        valid is False the slot is not read, is not
                                        counted, and comes back as zeros.
        store.n                         S
    idx: integers, shape (T, k). idx[t] lists the tokens query t attends to; -1 marks an
        unused slot.
    Returns float32, shape (T, H, d): per query and head, the softmax over the selected
    tokens of q . key * d**-0.5, applied to their values. A query with no selected
    token returns zeros.
    """
    raise NotImplementedError
