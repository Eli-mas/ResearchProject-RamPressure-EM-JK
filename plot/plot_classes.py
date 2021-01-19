from pathlib import Path

from mpl_wrap.plot_classes import *
from common import consume

from matplotlib.projections import register_projection
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt

from .plotting_functions import paper_tex_save, misc_fig_save, keepax


class PaperFigure(Figure):
	def __init__(self, paper=None, *a, **k):
		Figure.__init__(self, *a, **k)
		
		self.set_paper(paper)
	
	def set_paper(self, paper):
		self.paper = paper
	
	def save(self, name, link_folder=None, close=False, **k):
		if self.paper is None:
			raise ValueError(
				'the instance attribute `paper` has not been set on this figure; '
				'call `set_paper` or provide in initialization')
		if not link_folder:
			return misc_fig_save(name, self.paper, fig=self, **k)
		else:
			return paper_tex_save(name, self.paper,
						   link_folder=link_folder, close=close, fig=self, **k)

class BindingPaperFigure(PaperFigure, BindingFigure):
	def __init__(self, *a, paper=None, **kw):
		BindingFigure.__init__(self, *a, **kw)
		PaperFigure.__init__(self, *a, paper=paper, **kw)


if __name__=='__main__':
	fig = plt.figure()
# 	register_projection(BoundedAxes)
	bounder = plt.subplot()
	ax = fig.add_axes(bounder.get_position(),projection = 'bounded')
	ax.set_bounder(bounder)
	bounder.axis([10,20,30,40])
	bounder.axis('off')
	plt.show()

from mpl_wrap.plot_classes import __all__
__all__ = ('PaperFigure', 'BindingPaperFigure', *__all__)