import numpy as np


def print_rules(a, rel_dict):
    rules = []
    for k, v in rel_dict.iteritems():
        query_matrix = a[k]
        rules.append([])
        for rule in range(query_matrix.shape[0]):
            body_atoms = query_matrix[rule]
            body_atoms_ordered = np.argsort(body_atoms)
            weights = np.sort(body_atoms)
            for atom_index, weight in zip(body_atoms_ordered, weights):
                if rule == 0:
                    rules[k].append(rel_dict[k] + '<-' + str(weight) + ' ' + rel_dict[atom_index])
                else:
                    rules[k].append((' /\ ' + str(weight) + ' ' + rel_dict[atom_index]))
    print(rules)

if __name__ == '__main__':
    a = np.random.random((3, 2, 3))
    rel_dict = {0: 'a', 1: 'b', 2: 'c'}
    print_rules(a, rel_dict)