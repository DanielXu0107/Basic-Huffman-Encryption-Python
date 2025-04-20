"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    result = {}
    for byte in text:
        if byte in result:
            result[byte] += 1
        else:
            result[byte] = 1
    return result


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    trees = []
    for symbol in freq_dict:
        trees.append((freq_dict[symbol], HuffmanTree(symbol)))

    if len(trees) == 1:
        symbol = trees[0][1].symbol
        dummy = (symbol + 1) % 256
        return HuffmanTree(None, HuffmanTree(symbol), HuffmanTree(dummy))

    while len(trees) > 1:
        sort_trees(trees)
        freq1, tree1 = trees.pop(0)
        freq2, tree2 = trees.pop(0)
        merged_tree = HuffmanTree(None, tree1, tree2)
        trees.append((freq1 + freq2, merged_tree))

    return trees[0][1]


def sort_trees(trees: list[tuple[int, HuffmanTree]]) -> None:
    """Sort the list of (frequency, tree) tuples in-place by frequency."""
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i][0] > trees[j][0]:
                trees[i], trees[j] = trees[j], trees[i]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanTree(None, \
                    HuffmanTree(None, HuffmanTree(3), HuffmanTree(2)),\
                    HuffmanTree(None, HuffmanTree(1), HuffmanTree(4)))
    >>> x = get_codes(tree)
    >>> x = {3: "00", 2: "01", 1: "10", 4: "11"}

    """
    if tree.is_leaf():
        return {tree.symbol: ""}

    codes = {}

    if tree.left:
        left_codes = get_codes(tree.left)
        for symbol in left_codes:
            codes[symbol] = '0' + left_codes[symbol]

    if tree.right:
        right_codes = get_codes(tree.right)
        for symbol in right_codes:
            codes[symbol] = '1' + right_codes[symbol]

    return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.number
    2
    >>> tree.right.number
    1
    """
    _assign_numbers(tree, [0])


def _assign_numbers(node: HuffmanTree, counter: list[int]) -> None:
    if node.left:
        _assign_numbers(node.left, counter)
    if node.right:
        _assign_numbers(node.right, counter)

    if node.left or node.right:
        node.number = counter[0]
        counter[0] += 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    codes = get_codes(tree)

    weighted_sum =\
        sum(len(codes[symbol]) * freq_dict[symbol] for symbol in freq_dict)

    total_freq = sum(freq_dict.values())

    return weighted_sum / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bit_str = ''.join(codes[byte] for byte in text)
    result = []

    for i in range(0, len(bit_str), 8):
        byte_bits = bit_str[i:i + 8]
        result.append(bits_to_byte(byte_bits))

    return bytes(result)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1,
    0, 100, 0, 111, 0, 108, 1, 3, 1, 2, 1, 4]
    """
    if tree.is_leaf():
        return b''

    result = []

    result.extend(tree_to_bytes(tree.left))
    result.extend(tree_to_bytes(tree.right))

    if tree.left.is_leaf():
        result.append(0)
        result.append(tree.left.symbol)
    else:
        result.append(1)
        result.append(tree.left.number)

    if tree.right.is_leaf():
        result.append(0)
        result.append(tree.right.symbol)
    else:
        result.append(1)
        result.append(tree.right.number)

    return bytes(result)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    node = node_lst[root_index]
    if node.l_type == 0:
        left = HuffmanTree(node.l_data, None, None)
    else:
        left = generate_tree_general(node_lst, node.l_data)
    if node.r_type == 0:
        right = HuffmanTree(node.r_data, None, None)
    else:
        right = generate_tree_general(node_lst, node.r_data)
    return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    stack = []

    for i in range(root_index + 1):  # Traverse through postorder list
        node = node_lst[i]

        # Right child
        if node.r_type == 0:
            right = HuffmanTree(node.r_data)
        else:
            right = stack.pop()

        # Left child
        if node.l_type == 0:
            left = HuffmanTree(node.l_data)
        else:
            left = stack.pop()

        # Combine into internal node
        combined = HuffmanTree(None, left, right)
        stack.append(combined)

    return stack[0]


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    result = []
    s1 = "".join([byte_to_bits(byte) for byte in text])
    curr = tree
    for binary in s1:
        if binary == "0":
            curr = curr.left
        elif binary == "1":
            curr = curr.right
        if curr.is_leaf():
            result.append(curr.symbol)
            curr = tree
            if len(result) == size:
                break
    return bytes(result)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    symbol_freqs = []
    for symbol in freq_dict:
        symbol_freqs.append((symbol, freq_dict[symbol]))

    for i in range(len(symbol_freqs)):
        for j in range(i + 1, len(symbol_freqs)):
            if symbol_freqs[i][1] < symbol_freqs[j][1]:
                symbol_freqs[i], symbol_freqs[j] =\
                    symbol_freqs[j], symbol_freqs[i]

    # Step 2: Traverse tree and collect all leaf nodes (using a stack)
    leaf_nodes = []
    stack = [tree]

    while stack:
        node = stack.pop()
        if node.left is None and node.right is None:
            leaf_nodes.append(node)
        else:
            # Push right first so left is processed first
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    # Step 3: Assign most frequent symbols to shallowest leaf positions
    for i in range(len(leaf_nodes)):
        leaf_nodes[i].symbol = symbol_freqs[i][0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
