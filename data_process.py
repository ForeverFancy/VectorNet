import numpy as np
import pickle


OBJ_MAP = {"AGENT": 0, "AV": 1, "OTHERS": 2}


def load_features(paths: list):
    '''
    load features from paths

    @input paths (list): path to trag_list

    @return features (np.ndarray) of shape (num_of_paths, maxnum_of_global_nodes, maxnum_of_sub_nodes, 7): feature of all subgraph, padding to maxnum_of_seqs

    @return subgraph_mask (np.ndarray) of shape (len(feature_list), maxnum_of_global_nodes, maxnum_of_sub_nodes): mask of subgraph nodes

    @return attention_mask (np.ndarray) of shape (len(feature_list), maxnum_of_global_nodes): attention mask of padding nodes

    @return groundtruth (np.ndarray): groundtruth for prediction

    @return groundtruth (np.ndarray): mask for real groundtruth (split padding)
    '''
    groundtruth_list = []
    padding_features_list = []
    mask_list = []
    agent_index = None
    for path in paths:
        with open(path, "rb") as f:
            traj_list = pickle.load(f)
        features = []
        for i in range(len(traj_list)):
            vec, ground_truth = build_vector(traj_list[i], i)
            # print(vec.shape)
            features.append(vec)
            if ground_truth is not None:
                groundtruth_list.append(ground_truth)
                agent_index = i

        padding_features, mask = padding_trajectory(features, agent_index)
        # print(padding_features.shape, mask.shape)
        padding_features_list.append(padding_features)
        mask_list.append(mask)
    # print(agent_index)
    features, subgraph_mask, attention_mask = global_padding(padding_features_list, mask_list)
    # print(features.shape, subgraph_mask.shape, attention_mask.shape)

    groundtruth, groundtruth_mask = handle_ground_truth(groundtruth_list)
    return features, subgraph_mask, attention_mask, groundtruth, groundtruth_mask


def build_vector(traj: np.ndarray, id: int):
    '''
    build vectors based on the input trajectory

    @input traj (np.ndarray): traject of one object

    @input id (int): j in the paper, integer id of P_j, indicating v_i is in P_j 

    @return vector (np.ndarray) of shape (len(traj) - 1, 7): vector build by trajectory, each row contains (x_start, y_start, x_end, y_end, obj_type, time_stamp, j)

    @return ground_truth (np.ndarray): return groundtruth trajectory if the input is an agent, otherwise return None
    '''
    # print(len(tarj))
    # print(tarj)
    ground_truth = None
    vector = np.zeros((len(traj) - 1, 7))

    # start coordinates (x_start, y_start)
    vector[:, 0] = traj[:, 3][:-1]
    vector[:, 1] = traj[:, 4][:-1]

    # end coordinates (x_end, y_end)
    vector[:, 2] = traj[:, 3][1:]
    vector[:, 3] = traj[:, 4][1:]

    # obj_type, time_stamp, j
    vector[:, 4] = OBJ_MAP[traj[0, 2]]
    vector[:, 5] = traj[:, 0][:-1]
    vector[:, 6] = id

    if traj[0, 2] == "AGENT":
        ground_truth = vector[np.where(vector[:, 5] > 2), :].squeeze(axis=0)
        vector = vector[np.where(vector[:, 5] <= 2), :].squeeze(axis=0)
        # print(vector.shape, ground_truth.shape)

    # print(vector)
    return vector, ground_truth


def padding_trajectory(features: list, agent_index: int, max_seq_length: int = 49):
    '''
    Padding the input features to max sequence length (max number of sub nodes), and swap agent index to 0

    @input features: raw features

    @input agent_index (int): if not None, swap agent feature to index 0

    @input max_seq_length: padding to max sequence length (num of sub nodes), default is 49 (5 sec, 0.1 sec sampling)

    @return padding_features of shape (len(features), maxnum_of_sub_nodes, 7)

    @return mask of shape (len(features), maxnum_of_sub_nodes, 7)
    '''
    if agent_index is not None:
        # Swap agent index to 0
        tmp = features[0]
        features[0] = features[agent_index]
        features[agent_index] = tmp
        # print(features[0].shape, features[agent_index].shape)

    seq_length = [x.shape[0] for x in features]
    # print(seq_length)
    max_seq_length = max(
        seq_length) if max_seq_length is not None else max_seq_length
    mask = np.zeros((len(features), max_seq_length))
    padding_features = np.zeros((len(features), max_seq_length, 7))
    for i, feature in enumerate(features):
        mask[i, : feature.shape[0]] = 1
        mask[i, feature.shape[0]:] = 0
        padding_features[i, :, :] = np.concatenate(
            (feature, np.zeros((max_seq_length - feature.shape[0], 7))), axis=0)
    # print(padding_features, mask)
    return padding_features, mask


def global_padding(feature_list: list, mask_list: list):
    '''
    padding all trajectories to the same number of nodes

    @input feature_list (list): input feature list

    @input mask_list (list): subgraph mask list

    @return features (np.ndarray) of shape (len(feature_list), maxnum_of_global_nodes, maxnum_of_sub_nodes, 7)

    @return subgraph_mask (np.ndarray) of shape (len(feature_list), maxnum_of_global_nodes, maxnum_of_sub_nodes)

    @return attention_mask (np.ndarray) of shape (len(feature_list), maxnum_of_global_nodes)
    '''
    assert len(feature_list) == len(mask_list)

    length = len(feature_list)
    num_of_seqs = [feature.shape[0] for feature in feature_list]
    maxnum_of_seqs = max(num_of_seqs)
    # print(mask_list[0].shape)
    features = np.zeros((len(feature_list), maxnum_of_seqs,
                         feature_list[0].shape[1], feature_list[0].shape[2]))
    attention_mask = np.zeros((len(feature_list), maxnum_of_seqs))
    subgraph_mask = np.zeros((len(feature_list), maxnum_of_seqs, feature_list[0].shape[1]))
    # print(features.shape)
    for i, f in enumerate(feature_list):
        features[i, :, :, :] = np.concatenate(
            (f, np.zeros((maxnum_of_seqs - f.shape[0], f.shape[1], f.shape[2]))), axis=0)
        attention_mask[i, : f.shape[0]] = 1
        subgraph_mask[i, : f.shape[0], :] = mask_list[i]
    # print(features, subgraph_mask , attention_mask)
    return features, subgraph_mask, attention_mask


def handle_ground_truth(groundtruth_list: list, max_groundtruth_length: int = 30):
    '''
    @input groundtruth_list (list): input groundtruth

    @input max_groundtruth_length (int): maximum length of groundtruth, default is 30 (padding to the same length)

    @return groundtruth (np.ndarray) of shape (len(groundtruth_list), max_groundtruth_length * 4): each contains (x_start, y_start, x_end, y_end) 

    @return groundtruth_mask (np.ndarray) of shape (len(groundtruth_list), max_groundtruth_length * 4): mask where is not padding
    '''
    groundtruth_length = [gt.shape[0] for gt in groundtruth_list]
    max_groundtruth_length = max(groundtruth_length) if max_groundtruth_length is None else max_groundtruth_length
    groundtruth = np.zeros((len(groundtruth_list), max_groundtruth_length * 4))
    groundtruth_mask = np.zeros((len(groundtruth_list), max_groundtruth_length * 4))
    # print("-" * 80)
    # print(groundtruth_list)
    for i, gt in enumerate(groundtruth_list):
        groundtruth[i, : gt.shape[0] * 4] = gt[:, :4].reshape(-1, 1).squeeze(axis=1)
        groundtruth_mask[i, : gt.shape[0] * 4] = 1
    # print(groundtruth)
    return groundtruth, groundtruth_mask
        # groundtruth[i, :] = gt


if __name__ == "__main__":
    load_features(paths=["2645.save", "3700.save"])
