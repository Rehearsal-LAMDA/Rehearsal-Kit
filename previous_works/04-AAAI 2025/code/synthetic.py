import numpy as np


class Synthetic:
    def __init__(self, prob=0.2, seed=2024, dims=None):
        self.rng = np.random.default_rng(seed)
        if dims is None:
            self.x_dim = self.rng.choice([2, 4, 6, 8, 10], size=1)[0]
            self.z_dim = self.rng.choice([2, 4, 6, 8, 10], size=1)[0]
            self.y_dim = self.rng.choice([1, 2], size=1)[0]
        else:
            self.x_dim, self.z_dim, self.y_dim = dims
        self.n_vars = self.x_dim + self.z_dim + self.y_dim
        var2group = [0] * self.x_dim + [1] * self.z_dim + [2] * self.y_dim
        self.pairs = []
        adj = np.triu(self.rng.choice([0, 1], p=[1 - prob, prob], size=(self.n_vars, self.n_vars)), k=1)
        for child in range(self.n_vars - 1, -1, -1):
            parents = list(np.nonzero(adj[:, child])[0])
            if len(parents) > 0:
                if self.rng.random() < prob and len(self.pairs) > 0 and var2group[child] == var2group[child + 1]:
                    self.pairs.pop()
                    self.pairs.append(([child, child + 1], parents))
                else:
                    self.pairs.append(([child], parents))

    def get_children_parents_pairs(self):
        return self.pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars


class Syn1:
    def __init__(self):
        syn = Synthetic(0.3, 0)
        self.env_seed = 0
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [10, 11, 12, 17]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([0, 1, 4, -2])
        self.alter_ranges = [-3, 3]

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_alter_ranges(self):
        return self.alter_ranges


class Syn2:
    def __init__(self):
        syn = Synthetic(0.3, 4)
        self.env_seed = 4
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [12, 13, 14]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([4, -3, 2, 0])
        self.alter_ranges = [-3, 3]

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_alter_ranges(self):
        return self.alter_ranges


class MLP1:
    def __init__(self):
        syn = Synthetic(0.2, 0, [2, 8, 2])
        self.env_seed = 0
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [5, 6, 7, 8, 9]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([-15, 20, -7, 10])
        self.alter_ranges = [-2, 2]

    def get_noise(self):
        return 1.0

    def get_free_var_scale(self):
        return 0.1

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_standard_alter_ranges(self):
        return self.alter_ranges


class MLP2:
    def __init__(self):
        syn = Synthetic(0.15, 0, [1, 10, 2])
        self.env_seed = 2024
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [2, 5, 6, 7, 8]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([-5, 10, 5, 0])
        self.alter_ranges = [-1, 1]

    def get_noise(self):
        return 1.0

    def get_free_var_scale(self):
        return 0.1

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_standard_alter_ranges(self):
        return self.alter_ranges


class MLP4:
    def __init__(self):
        syn = Synthetic(0.15, 0, [8, 14, 2])
        self.env_seed = 2024
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [9, 13, 15, 20]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([0, 5, 5, 0])
        self.alter_ranges = [-2, 2]

    def get_noise(self):
        return 1.0

    def get_free_var_scale(self):
        return 0.1

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_standard_alter_ranges(self):
        return self.alter_ranges


class MLP3:
    def __init__(self):
        syn = Synthetic(0.15, 0, [2, 12, 2])
        self.env_seed = 0
        self.x_dim, self.z_dim, self.y_dim, self.n_vars = syn.get_info()
        self.children_parents_pairs = syn.get_children_parents_pairs()
        self.alterable_vars = [4, 5, 7, 9, 10, 12]
        self.M = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self.d = np.array([0, 5, 5, 0])
        self.alter_ranges = [-2, 2]

    def get_noise(self):
        return 1.0

    def get_free_var_scale(self):
        return 0.1

    def get_children_parents_pairs(self):
        return self.children_parents_pairs

    def get_info(self):
        return self.x_dim, self.z_dim, self.y_dim, self.n_vars

    def get_desired_region(self):
        return self.M, self.d

    def get_env_seed(self):
        return self.env_seed

    def get_alterable_vars(self):
        return self.alterable_vars

    def get_standard_alter_ranges(self):
        return self.alter_ranges
