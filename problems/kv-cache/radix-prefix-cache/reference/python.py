import heapq


class Node:
    """One edge of the tree and the node it leads to."""

    def __init__(self, parent: "Node | None", key: list[int], last_access: int):
        self.parent = parent
        self.key = key  # the tokens on the edge from the parent; empty only for the root
        self.children: dict[int, Node] = {}  # first token of a child's key -> child
        self.locks = 0  # running requests whose prefix passes through this node
        self.last_access = last_access


class RadixCache:
    def __init__(self):
        self._root = Node(None, [], 0)
        self._clock = 0  # one tick per match_prefix or insert
        self._tokens = 0

    def _walk(self, tokens: list[int]) -> tuple[int, Node]:
        """Follow `tokens` from the root as far as the tree goes.

        Returns the matched length and the node that ends exactly there, splitting
        an edge if the match ends inside it. Every node whose key was matched in
        full, including the upper half of a split, gets the current time.
        """
        self._clock += 1
        node, matched = self._root, 0
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            key = child.key
            common = 1  # the first token matched: that is how the child was found
            limit = min(len(key), len(tokens) - matched)
            while common < limit and key[common] == tokens[matched + common]:
                common += 1
            if common < len(key):
                child = self._split(child, common)
            child.last_access = self._clock
            node, matched = child, matched + common
            if common < len(key):
                break  # the match ended inside the old edge
        return matched, node

    def _split(self, child: Node, length: int) -> Node:
        """Cut the edge into `child` after `length` tokens; return the new upper node.

        `child` keeps its identity and its subtree and becomes the lower half, so a
        handle to it still stands for the same prefix. The upper half inherits the
        lock count (every lock on the lower half also holds the tokens above it) and
        the access time.
        """
        upper = Node(child.parent, child.key[:length], child.last_access)
        upper.locks = child.locks
        child.parent.children[upper.key[0]] = upper
        child.key = child.key[length:]
        child.parent = upper
        upper.children[child.key[0]] = child
        return upper

    def match_prefix(self, tokens: list[int]) -> tuple[int, Node]:
        return self._walk(tokens)

    def insert(self, tokens: list[int]) -> int:
        matched, node = self._walk(tokens)
        if matched < len(tokens):
            leaf = Node(node, list(tokens[matched:]), self._clock)
            node.children[leaf.key[0]] = leaf
            self._tokens += len(leaf.key)
        return matched

    def lock(self, node: Node) -> None:
        while node is not self._root:
            node.locks += 1
            node = node.parent

    def unlock(self, node: Node) -> None:
        while node is not self._root:
            node.locks -= 1
            node = node.parent

    def evict(self, n_tokens: int) -> int:
        # Only an unlocked leaf may go: an inner node is the prefix of its children.
        # Collect the candidates once, oldest on top of a heap.
        heap = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node is not self._root and node.locks == 0:
                heap.append((node.last_access, id(node), node))
        heapq.heapify(heap)
        evicted = 0
        while evicted < n_tokens and heap:
            _, _, victim = heapq.heappop(heap)
            parent = victim.parent
            del parent.children[victim.key[0]]
            evicted += len(victim.key)
            # A parent that just lost its last child is a leaf now, and a candidate.
            if not parent.children and parent is not self._root and parent.locks == 0:
                heapq.heappush(heap, (parent.last_access, id(parent), parent))
        self._tokens -= evicted
        return evicted

    def total_tokens(self) -> int:
        return self._tokens

    def nodes(self) -> list[tuple[tuple[int, ...], int, int]]:
        out = []
        stack = [(self._root, ())]
        while stack:
            node, prefix = stack.pop()
            for child in node.children.values():
                path = prefix + tuple(child.key)
                out.append((path, len(child.key), child.locks))
                stack.append((child, path))
        return sorted(out)
