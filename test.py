from __future__ import annotations

import time
from distutils.command.build import build
from tabnanny import check

from huffman import HuffmanTree
from utils import *
from compress import *


from huffman import HuffmanTree

def test_improve_tree():
    # Tests if improve_tree produces the expected improved tree.
    left = HuffmanTree(None, HuffmanTree(99, None, None), HuffmanTree(100, None, None))
    right = HuffmanTree(None, HuffmanTree(101, None, None),
                        HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    tree = HuffmanTree(None, left, right)
    freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    improve_tree(tree, freq)

    # After improve_tree, we expect 97 and 98 to be on shallower nodes than 100 and 101
    # This will fail because the function doesn't ensure minimal avg_length
    assert avg_length(tree, freq) < 2.49

def test_improve_tree_no_swaps():
    # Already optimal tree; nothing should change
    tree = HuffmanTree(None,
                       HuffmanTree(97, None, None),
                       HuffmanTree(None, HuffmanTree(98, None, None), HuffmanTree(99, None, None)))
    freq = {97: 10, 98: 5, 99: 1}
    before = avg_length(tree, freq)
    improve_tree(tree, freq)
    after = avg_length(tree, freq)
    assert before == after  # Shouldn't improve, but your function reassigns anyway

def test_improve_tree_single_swap():
    # Needs only 1 swap but current function reassigns all leaves
    tree = HuffmanTree(None,
                       HuffmanTree(97, None, None),
                       HuffmanTree(98, None, None))
    freq = {97: 1, 98: 10}
    improve_tree(tree, freq)
    # Should place 98 on left (shallower) but may not happen correctly
    assert tree.left.symbol == 98

def test_improve_tree_no_swaps_same_frequency():
    # All symbols have same freq, no swaps should occur
    tree = HuffmanTree(None,
                       HuffmanTree(97, None, None),
                       HuffmanTree(98, None, None))
    freq = {97: 5, 98: 5}
    before = [tree.left.symbol, tree.right.symbol]
    improve_tree(tree, freq)
    after = [tree.left.symbol, tree.right.symbol]
    assert before == after  # Shouldn’t change order

def test_improve_tree_only_valid_swaps_2():
    # Test 2: Ensure it doesn't put low-frequency symbol in a shallow position
    tree = HuffmanTree(None,
                       HuffmanTree(98, None, None),
                       HuffmanTree(99, None, None))
    freq = {98: 2, 99: 10}
    improve_tree(tree, freq)
    assert tree.left.symbol == 99



if __name__ == "__main__":
    test_improve_tree()
    test_improve_tree_only_valid_swaps_2()
    test_improve_tree_single_swap()
    test_improve_tree_no_swaps()
    test_improve_tree_no_swaps_same_frequency()
