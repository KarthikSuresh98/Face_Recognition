import numpy as np

#Generating ramdom pairs of images for face verification
def random_pairs(groups , size):
    out_img_a , out_img_b = [] , []
    all_groups = list(range(len(groups)))
    for match_group in [True , False]:
        group_idx = np.random.choice(all_groups , size = size)
        out_img_a += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
        else:
            non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
            b_group_idx = non_group_idx
        out_img_b += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in b_group_idx]  
    return np.stack(out_img_a , 0) , np.stack(out_img_b , 0)
