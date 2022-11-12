# import pickle

import torch
import torch.distributed as dist


class Comm(object):
    def __init__(self):
        self.local_rank = 0

    @property
    def world_size(self):
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @property
    def rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @property
    def local_rank(self):
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return self._local_rank

    @local_rank.setter
    def local_rank(self, value):
        if not dist.is_available():
            self._local_rank = 0
        if not dist.is_initialized():
            self._local_rank = 0
        self._local_rank = value

    @property
    def head(self):
        return 'Rank[{}/{}]'.format(self.rank, self.world_size)

    def is_main_process(self):
        return self.rank == 0

    def synchronize(self):
        if self.world_size == 1:
            return
        dist.barrier()


comm = Comm()


def all_gather(data):
    world_size = comm.world_size
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    world_size = comm.world_size
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_tensors(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(comm.world_size)
    ]

    dist.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather[comm.rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output
