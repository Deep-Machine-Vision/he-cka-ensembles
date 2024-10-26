""" GroupNorm for MLPs """

# @TODO do some actual test the paper is for conv don't apply to MLP...
# ATTEMPT_GROUPS = [32, 24, 18, 16, 12, 10, 8, 6, 4, 2, 1]  # order to try and fit groups
# GOAL_CHAN_PER_GROUP = 16  # attempt to get at least 16 channels/features per group
# MIN_CHAN_PER_GROUP = 6  # don't go below this or headed to instance norm territory


def calculate_mlp_groupnorm_groups(num_features: int):
  """ Attempts to automatically calculate number of groups via guidelines from
  https://arxiv.org/pdf/1803.08494.pdf

  I am going to assume target of 32 groups/16 channels per group

  Args:
    num_features (int): the number of "channels" or features
  """
  if num_features > 300:
    if num_features % 6 == 0:
      return 6
    if num_features % 5 == 0:
      return 5
  if num_features > 200:
    if num_features % 4 == 0:
      return 4
    if num_features % 3 == 0:
      return 3
  if num_features > 50:
    if num_features % 2 == 0:
      return 2
  return 1
