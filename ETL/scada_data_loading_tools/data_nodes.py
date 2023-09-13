from .abstract_nodes import LeafNode, BranchNode


class Tag(LeafNode):
    pass


class Equipment(BranchNode):
    child_class = Tag


class Role(BranchNode):
    child_class = Equipment


class Step(BranchNode):
    child_class = Role


class Batch(BranchNode):
    child_class = Step


class RootNode(BranchNode):
    child_class = Batch

    def __init__(self, base_dir_path):
        super().__init__(name=base_dir_path, parent_nodes=tuple())
