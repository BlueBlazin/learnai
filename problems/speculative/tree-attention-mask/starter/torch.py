import torch


def tree_inputs(parents: torch.Tensor, prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention mask and position ids for a prefix followed by a flattened token tree.

    parents: integer tensor of shape (N,); parents[i] is the node index of node i's
        parent (always smaller than i), or -1 if node i continues the prefix directly.
    prefix_len: P >= 1, the number of prefix tokens in front of the tree.
    Returns (mask, position_ids):
    mask: bool tensor of shape (P + N, P + N), True where the row may attend to the
        column. Prefix rows are causal; the row of node i (index P + i) sees the whole
        prefix, itself, and its ancestors, and nothing else.
    position_ids: integer tensor of shape (P + N,): 0 .. P - 1 for the prefix, and
        P + depth for a node (depth 0 for a node whose parent is -1).
    """
    raise NotImplementedError


def verify_tree(
    parents: torch.Tensor, tokens: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[list[int], int]:
    """Greedy verification of a token tree: the accepted root path and the next token.

    parents: integer tensor of shape (N,), as for tree_inputs.
    tokens: integer tensor of shape (N,), the draft token of every node.
    target_argmax: integer tensor of shape (N + 1,); entry 0 is the target's greedy
        token after the prefix, entry i + 1 its greedy token after node i (given the
        prefix and the path to node i).
    Returns (path, next_token): the node indices of the accepted path from the top of
    the tree down, as a list of Python ints (possibly empty), and the target's greedy
    token after the last accepted node (after the prefix if the path is empty), as a
    Python int.
    """
    raise NotImplementedError
