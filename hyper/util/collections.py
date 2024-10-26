""" Useful collection methods like a defaultdict-like OrderedDict"""
from typing import List, Union, Dict
from collections import OrderedDict, Callable


class DefaultOrderedDict(OrderedDict):
  # Modified from Sources: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict, http://stackoverflow.com/a/6190500/562769
  def __init__(self, default_factory=None, *a, **kw):
    if default_factory is None:
      default_factory = DefaultOrderedDict  # by default assume self (an ordered default dict) for missing keys
    
    if (default_factory is not None and
      not isinstance(default_factory, Callable)):
      raise TypeError('first argument must be callable')
    OrderedDict.__init__(self, *a, **kw)
    self.default_factory = default_factory

  def __getitem__(self, key):
    try:
      return OrderedDict.__getitem__(self, key)
    except KeyError:
      return self.__missing__(key)

  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    self[key] = value = self.default_factory()
    return value

  def __reduce__(self):
    if self.default_factory is None:
      args = tuple()
    else:
      args = self.default_factory,
    return type(self), args, None, None, self.items()

  def copy(self):
    return self.__copy__()

  def __copy__(self):
    return type(self)(self.default_factory, self)

  def __deepcopy__(self, memo):
    import copy
    return type(self)(self.default_factory,
                      copy.deepcopy(self.items()))

  def __repr__(self):
      return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                              OrderedDict.__repr__(self))
  
  def get_subitems(self, keys):
    """ Returns the last sub-item in the key """
    if len(keys) > 0:
      # traverse keys
      base = self.__getitem__(keys[0])
      for ind in range(1, len(keys)):
        base = self.__getitem__(base[ind])
      return base
    return None

  def set_subitems(self, keys, value, items=None):
    """ Sets the last sub-item in the key list, will traverse existing keys """
    if items is None:
      items = self
    
    # set value or traverse and get
    if len(keys) == 1:
      items[keys[0]] = value
    elif len(keys) > 1:
      self.set_subitems(keys[1:], value, items=items.__getitem__(keys[0]))
    return None
  

def flatten_keys(obj: Union[dict, OrderedDict, DefaultOrderedDict], track=None, track_empty=None, prefix='', include_empty: bool=False):
  """ Returns a flattened view of the object in the correct order
  
  EX: {"l1": {"l2": {"l3": 10}}, "l1_o": 2}
  will have the following returned
  {"l1.l2.l3": 10, "l1_o": 2}
  """
  # we don't have output defined yet
  if track is None:
    track = OrderedDict()
    
    if include_empty:
      track_empty = OrderedDict()

  if obj is None:
    return None  # nothing to check
  elif isinstance(obj, (dict, DefaultOrderedDict, OrderedDict)):
    # otherwise its a dict to scan
    for key, item in obj.items():
      if isinstance(item, (dict, OrderedDict, DefaultOrderedDict)):
        if include_empty and len(item) == 0:
          if len(prefix) > 0:
            track_empty[f'{prefix}.{key}'] = item
          else:
            track_empty[key] = item
        flatten_keys(item, track=track, track_empty=track_empty, prefix=(f'{prefix}.{key}' if len(prefix) > 0 else key))
      else:
        if len(prefix) > 0:
          track[f'{prefix}.{key}'] = item
        else:
          track[key] = item
    
    if include_empty:
      return track, track_empty
    return track
  return obj  # otherwise nothing to do


def unflatten_keys(flat_dict: dict) -> dict:
    """
    Unflattens a dictionary with dot-separated keys into a nested dictionary.
    
    EX: {"l1.l2.l3": 10, "l1_o": 2} 
    will be unflattened to:
    {"l1": {"l2": {"l3": 10}}, "l1_o": 2}
    """
    result = OrderedDict()

    for compound_key, value in flat_dict.items():
        keys = compound_key.split('.')
        d = result
        for key in keys[:-1]:
            if key not in d:
                d[key] = OrderedDict()  # Create nested dict if key does not exist
            d = d[key]
        d[keys[-1]] = value  # Assign value to the deepest key

    return result


if __name__ == '__main__':
  print(flatten_keys(
    {'l1': {'l2': {'l3': 1}, 'l2_0': 2}, 'l_0': 3}
  ))