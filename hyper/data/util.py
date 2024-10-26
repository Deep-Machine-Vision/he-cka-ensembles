from torch.utils.data.distributed import DistributedSampler


def ddp_args(dataset, ddp=False, shuffle=True, drop_last=False):
  if ddp:
    return {
      'shuffle': False,
      'sampler': DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    }
  return {
    'shuffle': shuffle,
    'drop_last': drop_last
  }
