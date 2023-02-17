## Pseudocode codes

### TreeSearch Algorithm

```
TREE SEARCH(problem, strategy):
    Initialize the frontier using the initial state of problem
    Initialize the explored set to be empty
    while True
        if the frontier is empty then
            return failure
        choose a leaf node and remove it from the frontier
        if the node contains a goal state then 
            return the corresponding solution
        add the node to the explored set
        expand the chosen node, adding the resulting nodes to the frontier
        only if not in the frontier or explored set
```

### Monte Carlo Tree Search Algorithm

```
MCTS(problem, strategy):
    Initialize the root node with the initial state of problem
    while True
        leaf = tree policy(root)
        reward = default policy(leaf)
        backup(leaf, reward)

tree policy(root):
    node = root
    while node is not a terminal node
        if node is fully expanded
            node = best child(node, strategy)
        else
            return expand(node)
    return node

default policy(node):
    while node is not a terminal node
        node = random child(node)
    return the reward for node

backup(node, reward):
    while node is not null
        node.visits += 1
        node.reward += reward
        node = node.parent
```

### GraphSearch Algorithm

```
GRAPH SEARCH(problem, strategy):
    Initialize the frontier using the initial state of problem
    Initialize the explored set to be empty
    while True
        if the frontier is empty then
            return failure
        choose a leaf node and remove it from the frontier
        if the node contains a goal state then 
            return the corresponding solution
        add the node to the explored set
        expand the chosen node, adding the resulting nodes to the frontier
        only if not in the frontier or explored set
```