# hironaka.policy
A Policy is a function mapping an observation to an action. This is a wrapper class of such a function. The inside can be a neural network, a hardcoded strategy, or whatever.

Need to implement:
`
__init__
predict
`

Note that `input_preprocess_for_host` and `input_preprocess_for_agent` are merely helper functions for input preprocessing of list-based observations (e.g., class `ListPoints`).
Feel free to override them for different purposes.

## .NNPolicy
It wraps a neural network inside.
