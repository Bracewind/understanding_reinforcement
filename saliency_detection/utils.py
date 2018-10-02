import torch


def tensor_deepcopy(tensor):
    list_repr = []
    for i in range(len(tensor)):
        list_repr.append(tensor[i])
    return torch.Tensor(list_repr).cuda()
