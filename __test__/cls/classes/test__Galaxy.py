import numpy as np
from cls.adc import *

def test__getattrs(fname,*attrs):
	g = Galaxy(fname)
	a = (getattr(g,attr) for attr in attrs)
	check=True
	for attr,va,vg in zip(attrs,a,g.getattrs(*attrs)):
		try:
			if isinstance(va,np.ndarray): condition = np.array_equal(va,vg)
			else: condition = va==vg
			assert condition
		except AssertionError:
			print(f'{fname}: inequality in getattrs for attribute {attr}')
			check=False
	
	assert check


