import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EdgeDataSet(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.hf = h5py.File(file_path, 'r')
        self.keys = list(self.hf.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        group = self.hf.get(key)
        V_1 = torch.tensor(np.array(group.get("V_1")), dtype=torch.float32)
        V_2 = torch.tensor(np.array(group.get("V_2")), dtype=torch.float32)
        labels = np.array(group.get("labels"), dtype=np.int64)
        labels = torch.from_numpy(labels)
        E_1_idx = np.array(group.get("E_1_idx")).transpose()
        E_1_values = np.array(group.get("E_1_values"))
        E_1_shape = np.array(group.get("E_1_shape"))

        E_1_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_1_idx), torch.from_numpy(E_1_values),
                                             E_1_shape.tolist())
        E_1 = E_1_sparse.to_dense()
        # E_1 = tf.Variable(tf.sparse.to_dense(E_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_1")

        E_2_idx = np.array(group.get("E_2_idx")).transpose()
        E_2_values = np.array(group.get("E_2_values"))
        E_2_shape = np.array(group.get("E_2_shape"))
        E_2_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_2_idx), torch.from_numpy(E_2_values),
                                             E_2_shape.tolist())
        E_2 = E_2_sparse.to_dense()

        E_3_idx = np.array(group.get("E_3_idx")).transpose()
        E_3_values = np.array(group.get("E_3_values"))
        E_3_shape = np.array(group.get("E_3_shape"))
        E_3_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_3_idx), torch.from_numpy(E_3_values),
                                             E_3_shape.tolist())
        E_3 = E_3_sparse.to_dense()

        A_2_idx = np.array(group.get("A_2_idx")).transpose()
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = torch.sparse_coo_tensor(torch.from_numpy(A_2_idx), torch.from_numpy(A_2_values),
                                             A_2_shape.tolist())
        A_2 = A_2_sparse.to_dense()

        A_3_idx = np.array(group.get("A_3_idx")).transpose()
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = torch.sparse_coo_tensor(torch.from_numpy(A_3_idx), torch.from_numpy(A_3_values),
                                             A_3_shape.tolist())
        A_3 = A_3_sparse.to_dense()

        return [V_1, E_1, E_2, E_3, V_2, A_2, A_3], labels
        pass


def dataloader_edge(file_path):
    hf = h5py.File(file_path, 'r')
    keys = list(hf.keys())
    print(len(keys))
    for key in list(hf.keys()):
        group = hf.get(key)

        V_1 = torch.tensor(np.array(group.get("V_1")), dtype=torch.float32)
        V_2 = torch.tensor(np.array(group.get("V_2")), dtype=torch.float32)
        labels = np.array(group.get("labels"), dtype=np.int64)
        labels = torch.from_numpy(labels)
        E_1_idx = np.array(group.get("E_1_idx")).transpose()
        E_1_values = np.array(group.get("E_1_values"))
        E_1_shape = np.array(group.get("E_1_shape"))

        E_1_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_1_idx), torch.from_numpy(E_1_values),
                                             E_1_shape.tolist())
        E_1 = E_1_sparse.to_dense()
        # E_1 = tf.Variable(tf.sparse.to_dense(E_1_sparse, default_value=0.), dtype=tf.dtypes.float32, name="E_1")

        E_2_idx = np.array(group.get("E_2_idx")).transpose()
        E_2_values = np.array(group.get("E_2_values"))
        E_2_shape = np.array(group.get("E_2_shape"))
        E_2_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_2_idx), torch.from_numpy(E_2_values),
                                             E_2_shape.tolist())
        E_2 = E_2_sparse.to_dense()

        E_3_idx = np.array(group.get("E_3_idx")).transpose()
        E_3_values = np.array(group.get("E_3_values"))
        E_3_shape = np.array(group.get("E_3_shape"))
        E_3_sparse = torch.sparse_coo_tensor(torch.from_numpy(E_3_idx), torch.from_numpy(E_3_values),
                                             E_3_shape.tolist())
        E_3 = E_3_sparse.to_dense()

        A_2_idx = np.array(group.get("A_2_idx")).transpose()
        A_2_values = np.array(group.get("A_2_values"))
        A_2_shape = np.array(group.get("A_2_shape"))
        A_2_sparse = torch.sparse_coo_tensor(torch.from_numpy(A_2_idx), torch.from_numpy(A_2_values),
                                             A_2_shape.tolist())
        A_2 = A_2_sparse.to_dense()

        A_3_idx = np.array(group.get("A_3_idx")).transpose()
        A_3_values = np.array(group.get("A_3_values"))
        A_3_shape = np.array(group.get("A_3_shape"))
        A_3_sparse = torch.sparse_coo_tensor(torch.from_numpy(A_3_idx), torch.from_numpy(A_3_values),
                                             A_3_shape.tolist())
        A_3 = A_3_sparse.to_dense()

        yield [V_1, E_1, E_2, E_3, V_2, A_2, A_3], labels

    hf.close()
    pass
