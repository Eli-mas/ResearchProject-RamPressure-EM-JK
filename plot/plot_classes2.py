from itertools import chain
from common import consume, ArgumentManager

import numpy as np

from matplotlib.projections import register_projection
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from cls.adc import Galaxy
from mpl_wrap.plot_classes import CustomAxes

from comp.polar_functions import polar_offset, polar_reduction
from comp.array_functions import ensure_elementwise

from prop.asy_prop_plot import COLORS, LABELS_LONG, MARKERS
from plot.plotting_functions import vline, full_legend

from.plot_classes import PaperFigure


ERROR_BAR_LW = 0.5

class _BaseAngleAxes(CustomAxes):
	""""""
	name = '_base_angle'
	ylim = np.array((0,1))
	xlim = np.array((-180,180))
	__defaults__ = {}
	
# 	def __new__(cls, *a, **kw):
# 		new = super(_BaseAngleAxes, cls).__new__(*a, **kw)
# 		new.custom_init = False
# 		return new
	
	def __init__(self, *a, **kw):
		CustomAxes.__init__(self, *a, **kw)
		self.stdev = {}
		self.height_map = {}
		self.plotted = False
# 		self.custom_init = True
	
	def add_angles(self, angles, angle_names, stdev=None, **kw):
		angles = ensure_elementwise(angles)
		if stdev is None:
			consume(self.add_angle(a, n, **kw)
					for a,n in zip(angles,angle_names))
		else:
			consume(self.add_angle(a, n, stdev = s, **kw)
					for a,n,s in zip(angles, angle_names, stdev))
		
	def set_height_map(self, h):
		self.height_map = h
	
	def keys(self): raise NotImplementedError(
		'_BaseAngleAxes does not implemente `keys` ;'
		f'implement in a subclass ({type(self)})'
	)
	
	def before_draw(self):
		try:
			if not self.plotted:
				self.axis([*self.xlim,*self.ylim])
				self.plot_angles()
# 				self.stdev.clear()
				self.set_yticks([])
				self.plotted=True
			else:
				print(f'{self}: before_draw called with self.plotted=True')
		except AttributeError:
			raise AttributeError("'self.plotted' is not defined; angles have not been initialized via 'add_angles'")
		else:
# 			vline(0,ax=self)
			pass
	
	def transform_args(self, args):
		return args

class _AngleMarkerAxes(_BaseAngleAxes):
	name='_angle_marker'
	
	__defaults__ = {'markersize':8.5}
	
	def __init__(self, *a, **kw):
		self.marker_manager = ArgumentManager()
		_BaseAngleAxes.__init__(self, *a, **kw)
	
	def keys(self): return self.marker_manager.keys()
	def values(self): return self.marker_manager.values()
	def items(self): return self.marker_manager.items()
	
	def add_angle(self, angle, name, stdev=None, **kw):
		if stdev is not None:
			self.stdev[name] = stdev
		
		self.marker_manager.add(name, angle, **kw)
	
	def plot_angles(self, text=None):
# 		print('`_AngleMarkerAxes.plot_angles` called')
		self.ensure_height_map()
		consume(self.marker_manager.add(n,self.height_map[n]) for n in self.marker_manager)
		
		for name, (a, k) in self.marker_manager.items():
			self.plot(*a, **_AngleMarkerAxes.__defaults__, **k)
		
		for name in (self.stdev.keys() & self.marker_manager.keys()):
			std = self.stdev.pop(name)
			(angle, y), kw = self.marker_manager[name]
			kw = kw.copy()
			kw.pop('marker')
			
			for adjust in (-360,0,360):
				self.plot([angle + adjust - std, angle + adjust + std],
						  [y, y], zorder=1, lw=ERROR_BAR_LW, **kw)
		
		if text is not None: self.text(.01, .95, text, size=15,va='top',
										ha='left',transform=self.transAxes)

class _AngleVlineAxes(_BaseAngleAxes):
	name='_angle_vline'
	
	__defaults__ = {'zorder':2,'ls':'--'}
	
	def __init__(self, *a, **kw):
		self.vline_manager = ArgumentManager()
		_BaseAngleAxes.__init__(self, *a, **kw)
	
	def keys(self): return self.vline_manager.keys()
	def values(self): return self.vline_manager.values()
	def items(self): return self.vline_manager.items()
	
	def add_angle(self, angle, name, stdev=None, **kw):
		if stdev is not None:
			self.stdev[name] = stdev
		
		self.vline_manager.add(name, angle, **kw)
	
	def plot_angles(self, text=None):
# 		print('`_AngleVlineAxes.plot_angles` called')
		self.ensure_height_map()
		
		for name, (a, k) in self.vline_manager.items():
			vline(*a, ax=self, **_AngleVlineAxes.__defaults__, **k)
		
		consume(self.vline_manager.add(n,self.height_map[n]) for n in self.vline_manager)
		for name in (self.stdev.keys() & self.vline_manager.keys()):
			std = self.stdev.pop(name)
			(angle, y), kw = self.vline_manager[name]
			
			for adjust in (-360,0,360):
				self.plot([angle + adjust - std, angle + adjust + std],
						  [y, y], lw=ERROR_BAR_LW, **_AngleVlineAxes.__defaults__, **kw)
		
		if text is not None: self.text(.01, .95, text, size=15,va='top',
										ha='left',transform=self.transAxes)

class _BaseReferenceAngleAxes:
	name='_base_reference_angle'
	_default_reference = 0
	
	def transform_args(self, args):
		"""First argument is an angle; transform it relative to
		the reference angle via polar_offset(angle, self.reference)"""
		return (polar_offset(args[0], self.reference), *args[1:])
	
	def set_reference(self, reference):
		self.reference = reference
	
	def ensure_reference(self, reference = _default_reference):
		if not hasattr(self,'reference'):
			self.reference = reference
	
	def assign_galaxy(self, filename):
		self.filename = filename

class _ReferenceAngleMarkerAxes(_BaseReferenceAngleAxes, _AngleMarkerAxes):
	name='_reference_angle_marker'
	
# 	def add_angles(self, angles, angle_names, **kw):
# 		super().add_angles(
# 			polar_functions.polar_offset(angles, self.reference),
# 			angle_names,
# 			**kw
# 		)
# 	
# 	def add_angle(self, angle, name, **kw):
# # 		print(f'{self.name}: adding {name}={angle} -- {kw}')
# 		super().add_angle(
# 			polar_functions.polar_offset(angle, self.reference),
# 			name,
# 			**kw
# 		)
	
	def plot_angles(self, text=None):
# 		print('`_AngleMarkerAxes.plot_angles` called')
		self.ensure_height_map()
		self.ensure_reference()
		consume(self.marker_manager.add(n,self.height_map[n]) for n in self.marker_manager)
		
		for name, ((angle, y), k) in self.marker_manager.items():
			self.plot(polar_offset(angle, self.reference), y,
					  **{**_AngleMarkerAxes.__defaults__, **k})
		
		for name in (self.stdev.keys() & self.marker_manager.keys()):
			std = self.stdev.pop(name)
			args, kw = self.marker_manager[name]
			angle,y = self.transform_args(args)
			kw = kw.copy()
			kw.pop('marker')
			
			for adjust in (-360,0,360):
				self.plot([angle + adjust - std, angle + adjust + std],
						  [y, y], zorder=1, lw=ERROR_BAR_LW, **kw)
		
		if text is not None: self.text(.01, .95, text, size=15,va='top',
										ha='left',transform=self.transAxes)

class _ReferenceAngleVlineAxes(_BaseReferenceAngleAxes, _AngleVlineAxes):
	name='_reference_angle_vline'
	
# 	def add_angles(self, angles, angle_names, **kw):
# 		super().add_angles(
# 			polar_functions.polar_offset(angles, self.reference),
# 			angle_names,
# 			**kw
# 		)
# 	
# 	def add_angle(self, angle, name, **kw):
# # 		print(f'{self.name}: adding {name}={angle} -- {kw}')
# 		super().add_angle(
# 			polar_functions.polar_offset(angle, self.reference),
# 			name,
# 			**kw
# 		)
	
	def plot_angles(self, text=None):
# 		print('`_AngleVlineAxes.plot_angles` called')
		self.ensure_height_map()
		self.ensure_reference()
		
		for name, (a, k) in self.vline_manager.items():
			vline(*self.transform_args(a), ax=self,
				  **{**_AngleVlineAxes.__defaults__, **k})
		
		consume(self.vline_manager.add(n,self.height_map[n]) for n in self.vline_manager)
		for name in (self.stdev.keys() & self.vline_manager.keys()):
			std = self.stdev.pop(name)
			args, kw = self.vline_manager[name]
			angle, y = self.transform_args(args)
			
			for adjust in (-360,0,360):
				self.plot([angle + adjust - std, angle + adjust + std],
						  [y, y], lw=ERROR_BAR_LW, **_AngleVlineAxes.__defaults__, **kw)
		
		if text is not None: self.text(.01, .95, text, size=15,va='top',
										ha='left',transform=self.transAxes)

class GalaxyAngleMarkerAxes(_ReferenceAngleMarkerAxes):
	"""class must be updated to handle asy/ref angles separately,
	as in GalaxyRefAxes"""
	
	name='galaxy_angle_marker'
	
	def add_angles(self, angle_names, **kw):
		angles = Galaxy.read_scalars(self.filename, angle_names)
		_ReferenceAngleMarkerAxes.add_angles(self, angles, angle_names, **kw)
		print(self.filename, self.angles)

class _GalaxyRefAxesBase(_ReferenceAngleMarkerAxes, _ReferenceAngleVlineAxes):
	name='_galaxy_ref_base'
	
	from prop.ref_rel_data import get_ref_data, ref_colors, ref_dict
	get_ref_data = staticmethod(get_ref_data)
	
	def __init__(self, *a, **kw):
		_ReferenceAngleMarkerAxes.__init__(self, *a, **kw)
		_ReferenceAngleVlineAxes.__init__(self, *a, **kw)
	
	def keys(self):
		k = set(_ReferenceAngleMarkerAxes.keys(self))
		k.update(_ReferenceAngleVlineAxes.keys(self))
		return k
	
	def ensure_height_map(self):
		k = self.keys()
		if ((not isinstance(self.height_map, dict)) or
			 self.height_map.keys() != k):
			heights = np.linspace(.1,.9, len(k))
			self.height_map = dict(zip(k, heights))
	
	def add_asy_angles(self, angle_names, **kw):
		angles = Galaxy.read_scalars(self.filename, angle_names)
		consume(self.add_asy_angle(n, a, **kw) for n,a in zip(angle_names, angles))
# 		print(self.filename, self.angles)
	
	def add_ref_angles(self):
		consume(self.add_ref_angle(rtype, (angle+180)%360, std)
				for rtype, angle, std in self.get_ref_data(self.filename))
	
	def add_asy_angle(self, name, value=None, std=None):
		if value is None:
			value = float(Galaxy.read_scalars(self.filename, name))
		if std is None:
# 			print('GalaxyRefAxes.add_asy_angle: getting std for', self.filename, name)
			try: std = float(Galaxy.get_noise_std(self.filename, name))
			except KeyError: pass
		_ReferenceAngleMarkerAxes.add_angle(
			self, value, name, color = COLORS[name], stdev=std,
			marker = MARKERS[name]#, label = LABELS_LONG[name]
		)
	
	@staticmethod
	def _ref_angle_defaults(name):
		value,std = self.ref_dict[self.filename][name]
		return (value + 180) % 360, std
		

class GalaxyRefAxes(_GalaxyRefAxesBase):
	name='galaxy_ref'
	
	def plot_angles(self):
# 		print('\t`GalaxyRefAxes.plot_angles` called')
		_ReferenceAngleMarkerAxes.plot_angles(self)
		_ReferenceAngleVlineAxes.plot_angles(self)
	
	def add_ref_angle(self, name, value=None, std=None):
		if value is None:
			value, std = self._ref_angle_defaults(name)
		
		_ReferenceAngleVlineAxes.add_angle(
			self, value, name, color = self.ref_colors[name],
			stdev = std# label = LABELS_LONG[name]
		)

class GalaxyRefAggAxes(_GalaxyRefAxesBase):
	name = 'galaxy_ref_agg'
	
	from prop.ref_rel_data import unique_ref_types, ref_colors
	
	ref_types = unique_ref_types()
	ref_marker_span, ref_marker_space = 5, 10
	ref_marker_start = 180 - ref_marker_space
	aggplot_ref_positions=dict(zip(
		ref_types,
		np.arange(ref_marker_start + ref_marker_span
					 - len(ref_types) * ref_marker_span,
				  ref_marker_start + ref_marker_span,
				  ref_marker_span)
	))
	aggplot_ref_markers=dict(zip(ref_types,('o','s','^','+','x')))
	
	del unique_ref_types, ref_marker_span, ref_marker_space, ref_marker_start
	
	def __init__(self, *a, **kw):
		self.ref_types = []
		self.ref_values = []
		super().__init__(*a, **kw)
	
	def add_ref_angle(self, name, value=None, std=None):
		if value is None:
			value,std = self._ref_angle_defaults(name)
		
		self.ref_types.append(name)
		self.ref_values.append((value, std))
	
	def plot_angles(self, *a, **kw):
# 		print(f'{self.__class__}: {self.filename}: self.ref_values:',self.ref_values)
		ref = polar_reduction(np.mean, [angle for angle,std in self.ref_values], 360)
		self.set_reference(ref)
		_ReferenceAngleMarkerAxes.plot_angles(self, *a, **kw)
		assert self.reference == ref
		vline(0, ls='--', c='k', lw=2, ax=self)
		for k in self.ref_types:
			self.scatter(self.aggplot_ref_positions[k], .5,
					   c = self.ref_colors[k], marker = self.aggplot_ref_markers[k])

class _GalaxyRefFigureBase(PaperFigure):
	_angles = ['EA','FA','HTA','qEA','TA','EA_trig']
	_filenames = ['4921','4501','4522','4254','4330','4402','4569','4388',]
	
	def __init__(self, filenames=_filenames, angles=_angles, *a, **kw):
		super().__init__(*a, **kw)
		gs = GridSpec(len(filenames)+1, 8, hspace=0, wspace=0)
		consume([ # implicitly defines `self.axes`
			self.add_subplot(gs[i,:-1], projection = self._gal_axes_class)
			for i in range(gs.get_geometry()[0]-1)
		])
		self.add_subplot(gs[-1,:-1])
		
		for a,f in zip(self.axes[:-1], filenames):
# 			a.set_reference(0)
# 			print('assigning galaxy:',f)
			a.assign_galaxy(f)
			a.add_asy_angles(angles)
			a.add_ref_angles()
			n = 'C' if f=='4921' else 'V'
			a.text(-180, .95, f'NGC {f} ({n})', fontsize=20, va='top', ha='left')
	
	def save(self, *a, legend_top = .75, **kw):
		self._format(legend_top)
		super().save(*a, **kw)
	
	def _format(self, legend_top):
		self._base_format(legend_top)
		self._ref_legend(legend_top)
	
	def _base_format(self, legend_top):
# 		self.axes[-1].axis('off')
		self.axes[-1].set_xticks(np.arange(-180,180+30,30))
		self.axes[-1].set_xlim(-180, 180)
		self.axes[-1].set_xlabel(self._xlabel,fontsize=18)
		
		for name,(_,kw) in self.axes[0].marker_manager.items():
			self.axes[-1].plot([],[], label=name,
							   **_AngleMarkerAxes.__defaults__, **kw)
		self.axes[-1].legend(
			ncol=len(self.axes[0].marker_manager), bbox_to_anchor=(.5,legend_top),
			fontsize=15, frameon=False, columnspacing=1, loc='center'
		)

class GalaxyRefFigure(_GalaxyRefFigureBase):
	_gal_axes_class = 'galaxy_ref'
	_xlabel = f'Angle from North (\N{DEGREE SIGN})'
# 	def asy_keys(self):
	
# 	def _ref_keys(self):
# 		return set(chain.from_iterable(
# 			ax.vline_manager.keys()
# 			for ax in self.axes
# 		))
	
	def _ref_kwargs(self):
		return dict(chain.from_iterable(
			ax.vline_manager.kwargs()
			for ax in self.axes[:-1]
		))
	
	def _ref_legend(self, legend_top):
		unique_refs_kwargs = self._ref_kwargs()
		leg_ax_alt = self.axes[-1].twinx()
		leg_ax_alt.axis('off')
		for name,kw in unique_refs_kwargs.items():
			leg_ax_alt.plot([],[], label=name,
						    **{**_AngleVlineAxes.__defaults__, **kw})
		
		leg_ax_alt.legend(
			ncol = len(unique_refs_kwargs), frameon = False,
			bbox_to_anchor=(.5, legend_top-.5),fontsize=15,loc='center',
			columnspacing=1, handletextpad = .2
		)

class GalaxyRefAggFigure(_GalaxyRefFigureBase):
	_gal_axes_class = 'galaxy_ref_agg'
	_xlabel = f'Angle relative to reference mean (\N{DEGREE SIGN})'
	from prop.ref_rel_data import ref_colors
	
	def _ref_legend(self, legend_top):
		leg_ax_alt = self.axes[-1].twinx()
		leg_ax_alt.axis('off')
		for k, m in self.axes[0].aggplot_ref_markers.items():
			leg_ax_alt.scatter([],[], label=k, c = self.ref_colors[k], marker = m)
		
		leg_ax_alt.legend(
			ncol = len(self.axes[0].aggplot_ref_markers), frameon=False,
			bbox_to_anchor=(.5,legend_top-.5),fontsize=15,loc='center',
			columnspacing=1, handletextpad = .2
		)

for c in (
	_AngleMarkerAxes, _ReferenceAngleMarkerAxes,
	_AngleVlineAxes, _ReferenceAngleVlineAxes,
	GalaxyAngleMarkerAxes, GalaxyRefAxes, GalaxyRefAggAxes
):
	register_projection(c)

if __name__=='__main__':
	from matplotlib import pyplot as plt
	
	fig = plt.figure(FigureClass = GalaxyRefFigure, paper='atlas')
	fig.save('ref comparisons', link_folder='reference data galaxies')
	fig = plt.figure(FigureClass = GalaxyRefAggFigure, paper='atlas')
	fig.save('ref comparisons agg', link_folder='reference data galaxies')
