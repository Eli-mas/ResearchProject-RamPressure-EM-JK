"""common arrays used in this project."""

import numpy as np
from .asy_prop import al,ahl,aql,a1hl,a2l, index_to_deg

range_aql=np.arange(aql)
range_ahl=np.arange(ahl)
range_al=np.arange(al)
range_a2l=np.arange(a2l)

range_a2l_deg=range_a2l*index_to_deg

birange_aql=np.arange(-aql,aql+1)
birange_ahl=np.arange(-ahl,ahl+1)

trig_weights=np.sin(np.linspace(0,np.pi,101))

__all__=('range_aql','range_ahl','range_al','range_a2l',
'range_a2l_deg','birange_aql','birange_ahl','trig_weights')