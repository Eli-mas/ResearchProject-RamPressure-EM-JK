from matplotlib import pyplot as plt
import numpy as np

from .plotting_functions import *
from .plot_classes import PaperFigure
from prop.asy_prop import *
from prop.asy_defaults import *

from common import consume

def m_component(m, *, t = range_a2l_rad, z=0, f=0, e=2, **kw):
	print(f)
	return f + (2 + np.sin(m*(t-z*deg_to_rad)))**e

@makeax_new('pax', func = ax_0N)
def plot_m_component(*a, pax = None, t = range_a2l_rad, **kw):
	p, = pax.plot(t, m_component(*a, t=t, **kw))
	rtext(pax, p)

def rtext(ax, line, size=16, **kw):
	ax.set_ylim(bottom=0, top=ax.axis()[-1]*1.25)
	
	r = line.get_data()[1]
	rlim = ax.axis()[-1]
	d = .5 * (rlim - r.max())
	for i in (0, .25, .5, .75):
		ax.text(tau*i, d+r[int(r.size * i)], int(r[int(r.size * i)]),
		 ha='center', va='center', size=size, **kw)
	
	ax.set_yticks(())

if __name__ == '__main__':
	fig, axes = plt.subplots(2,2, FigureClass = PaperFigure, paper='methods',
							 subplot_kw={'polar':True}, gridspec_kw=dict(wspace=-0.4))
	for i,(f,ax) in enumerate(zip((2,10,10,4), axes.flat)):
		if i<2:
			plot_m_component(2, z=45, f=f, pax=ax)
		elif i==2:
			p,=ax.plot(range_a2l_rad, .1*m_component(1, e=4) + m_component(2, z=45, f=f))
			rtext(ax, p)
		elif i==3:
			p,=ax.plot(
				range_a2l_rad,
				10*(-.2*m_component(1, z=90, f=f, e=2)
					+ m_component(2, z=45, f=f)
					+ .2*m_component(1, z=0, f=f, e=3)
					+ .6*m_component(1, z=90, f=f, e=3))
			)
			rtext(ax, p)
	
	consume(ax_0N(ax) for ax in axes.flat)
	
# 	plt.show()
	fig.save("m=2 variations", link_folder = '--misc')