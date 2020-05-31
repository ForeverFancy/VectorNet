import numpy as np
import pickle


OBJ_MAP = {"AGENT": 0, "AV": 1, "OTHERS": 2}


def load_features(path: str = './2645.save'):
    '''
    @input path (str): path to trag_list

    @return feature (list) shape of (nums_of_obj): feature of all subgraph, each item in list has shape (len(obj_traj) - 1, 7)
    '''
    with open(path, "rb") as f:
        traj_list = pickle.load(f)

    features = []
    for i in range(len(traj_list)):
        vec = build_vector(traj_list[i], i)
        # print(vec.shape)
        features.append(vec)
    padding_features, mask = padding_trajectory(features)
    return padding_features, mask


def build_vector(tarj: np.ndarray, id: int):
    '''
    @input tarj (np.ndarray): tarject of one object

    @input id (int): j in the paper, integer id of P_j, indicating v_i is in P_j 

    @return vector (np.ndarray) shape of (len(traj) - 1, 7): vector build by tarjectory, each row contains (x_start, y_start, x_end, y_end, obj_type, time_stamp, j)
    '''
    # print(len(tarj))
    # print(tarj)
    vector = np.zeros((len(tarj) - 1, 7))

    # start coordinates (x_start, y_start)
    vector[:, 0] = tarj[:, 3][:-1]
    vector[:, 1] = tarj[:, 4][:-1]

    # end coordinates (x_end, y_end)
    vector[:, 2] = tarj[:, 3][1:]
    vector[:, 3] = tarj[:, 4][1:]

    # obj_type, time_stamp, j
    vector[:, 4] = OBJ_MAP[tarj[0, 2]]
    vector[:, 5] = tarj[:, 0][:-1] - min(tarj[:, 0][:-1])
    vector[:, 6] = id

    # print(vector)
    return vector


def padding_trajectory(features: list):
    '''
    Padding the input features to max sequence length

    @input features: raw features

    @return padding_features(len(features), max_seq_length, 7)

    @return mask of shape(len(features), max_seq_length, 7)
    '''
    seq_length = [x.shape[0] for x in features]
    # print(seq_length)
    max_seq_length = max(seq_length)
    mask = np.zeros((len(features), max_seq_length))
    padding_features = np.zeros((len(features), max_seq_length, 7))
    for i, feature in enumerate(features):
        mask[i, : feature.shape[0]] = 1
        mask[i, feature.shape[0] :] = 0
        padding_features[i, :, :] = np.concatenate(
            (feature, np.zeros((max_seq_length - feature.shape[0], 7))), axis=0)
    # print(padding_features, mask)
    return padding_features, mask


if __name__ == "__main__":
    load_features()
