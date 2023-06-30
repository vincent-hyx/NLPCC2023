import copy
from copy import deepcopy
from queue import Queue
from treelib import Tree


class Element:
    def __init__(self, root_node, node, pos):
        self.root_node = root_node
        self.current_node = node
        self.pos = pos


class ASTreeNode:
    def __init__(self, left, right, parent, pos, data):
        self.left = left
        self.right = right
        self.parent = parent
        self.pos = pos
        self.data = data



def search_prefix(node_list, prefix_list):
    while len(node_list):
        point = node_list.pop()
        prefix_list.append(point.data)
        if point.left is not None:
            node_list.append(point.left)
            prefix_list = search_prefix(node_list, prefix_list)
        if point.right is not None:
            node_list.append(point.right)
            prefix_list = search_prefix(node_list, prefix_list)
    return prefix_list


def insert_prefix(prefix, parent=None):
    if len(prefix):
        token = prefix.pop()
        node = ASTreeNode(None, None, parent, 0, token)
        if token in "+-*/^":
            node.left, prefix = insert_prefix(prefix, node)
            node.right, prefix = insert_prefix(prefix, node)
        return node, prefix
    else:
        return None, None


def prefix_list2ast(prefix_list):
    prefix = deepcopy(prefix_list)
    prefix.reverse()
    tree, _ = insert_prefix(prefix)
    # search_prefix([tree])
    return tree


def transfer_add_mul(tree):
    temp = tree.left
    tree.left = tree.right
    tree.right = temp
    return tree


def equivalent_transfer(prefix_list):
    tree = prefix_list2ast(prefix_list)
    equivalent_eq = []
    stack = list()
    stack.append(tree)
    while len(stack):
        tree_node = stack.pop()
        # print(tree_node.data)
        if tree_node.data in "+*":
            t = copy.deepcopy(tree_node)
            temp = t.left
            t.left = t.right
            t.right = temp
            while t.parent is not None:
                t = t.parent
            equivalent_eq.append(search_prefix([t], []))
        if tree_node.left.data in "+-*/^":
            stack.append(tree_node.left)
        if tree_node.right.data in "+-*/^":
            stack.append(tree_node.right)
    return equivalent_eq



# def equivalent_transfer(prefix_list):
#     tree = prefix_list2ast(prefix_list)
#     equivalent_eq = []
#     stack = list()
#     stack.append(tree)
#     while len(stack):
#         tree_head_node = stack.pop()
#         t = copy.deepcopy(tree_head_node)
#         if t.data == '+':
#             temp = t.left
#             if temp.data == '-':
#                 if t.parent.left == t:
#                     t.parent.left = temp
#                     t.left = temp.right
#                     temp.right = t
#                     while t.parent is not None:
#                         t = t.parent
#                     equivalent_eq.append(search_prefix([t], []))
#                     # print(equivalent_eq)
#                 if t.parent.right == t:
#                     t.parent.right = temp
#                     t.left = temp.right
#                     temp.right = t
#                     while t.parent is not None:
#                         t = t.parent
#                     equivalent_eq.append(search_prefix([t], []))
#
#         if t.data == '/':
#             stack.append(t.left)







a = ['-', '*', "+", "20","5", "3", "10"]
# a = ["/", "+", "-", "N0", "*","N1" ,"N2", "N3", "N4"]
# b= ["+", "N1", "+", "N2", "+", "N3", "N4"]
print(equivalent_transfer(a))











