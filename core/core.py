from collections import deque
from itertools import islice


import inspect
from functools import partial, wraps
import numpy as np

from common import Struct, MultiIterator, IterDict
from common.parallel.threading import Queuer
from common.collections import consume, groupby_whole
from common.collections.array_funcs import assign as array_assign

def getattrs(obj,*attrs,asarray=True):
	if asarray: return np.array([getattr(obj,a) for a in attrs])
	else: return [getattr(obj,a) for a in attrs]

class AttributeAbsent(AttributeError):
	def __init__(self,galaxy,attr):
		self.attr = attr
		self.galaxy = str(galaxy)
	
	def __str__(self):
		return f"the attribute '{self.attr}' is absent on galaxy {self.galaxy}"
		#f"the attribute '{attr}' is not allowed for the galaxy '{self.filename}'"


__all__ = (
	'getattrs','MultiIterator', 'Struct', 'Queuer', 'consume',
	 'AttributeAbsent', 'groupby_whole', 'array_assign'
)

