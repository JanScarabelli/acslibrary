from .data_nodes import RootNode


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.root_node = RootNode(data_path)

    def iter_batches(self):
        yield from self.root_node.iter_child_names()

    def iter_steps(self, batch):
        yield from self.root_node[batch].iter_child_names()

    def iter_roles(self, batch, step):
        yield from self.root_node[batch][step].iter_child_names()

    def iter_equipments(self, batch, step, role):
        yield from self.root_node[batch][step][role].iter_child_names()

    def iter_tags(self, batch, step, role, equipment):
        yield from self.root_node[batch][step][role][equipment].iter_child_names()

    def load(self, batch=None, step=None, role=None, equipment=None, tag=None):
        batch_subset = [batch] if isinstance(batch, str) else batch
        step_subset = [step] if isinstance(step, str) else step
        role_subset = [role] if isinstance(role, str) else role
        equip_subset = [equipment] if isinstance(equipment, str) else equipment
        tag_subset = [tag] if isinstance(tag, str) else tag

        for batch in self.root_node.iter_children(batch_subset):
            for step in batch.iter_children(step_subset):
                for role in step.iter_children(role_subset):
                    for equip in role.iter_children(equip_subset):
                        for tag in equip.iter_children(tag_subset):
                            yield tag.load()
