# BUG: evict considers every unlocked node, not only leaves: removing an inner node drops its whole subtree, which is more than it counts and may be newer than what it should have removed.
import heapq


class Node:
    def __init__(self, parent: "Node | None", key: list[int], last_access: int):
        self.parent = parent
        self.key = key
        self.children: dict[int, Node] = {}
        self.locks = 0
        self.last_access = last_access


class RadixCache:
    def __init__(self):
        self._root = Node(None, [], 0)
        self._clock = 0
        self._tokens = 0

    def _walk(self, tokens: list[int]) -> tuple[int, Node]:
        self._clock += 1
        node, matched = self._root, 0
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            key = child.key
            common = 1
            limit = min(len(key), len(tokens) - matched)
            while common < limit and key[common] == tokens[matched + common]:
                common += 1
            if common < len(key):
                child = self._split(child, common)
            child.last_access = self._clock
            node, matched = child, matched + common
            if common < len(key):
                break
        return matched, node

    def _split(self, child: Node, length: int) -> Node:
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
        heap = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            if node is not self._root and node.locks == 0:
                heap.append((node.last_access, id(node), node))
        heapq.heapify(heap)
        evicted = 0
        while evicted < n_tokens and heap:
            _, _, victim = heapq.heappop(heap)
            parent = victim.parent
            if victim.key[0] not in parent.children or parent.children[victim.key[0]] is not victim:
                continue
            victim.parent = None
            del parent.children[victim.key[0]]
            evicted += len(victim.key)
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
