import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prop.simprop import vollmer_ranges
from numpy import nan
from prop.asy_prop_plot import color_darken

# INITIAL = 

"""
NGC4388: rps=0.4*1.25e17/((time-5.9e8)*(time-5.9e8)+(10.e7*10.e7))
NGC4501: rps=4.*1.28e17/((time-7.e8+3.e7)*(time-7.e8+3.e7)+(16.e7*16.e7))
NGC4330: rps=2.0*0.25*5.e17/((time-6.e8)*(time-6.e8)+(10.e7*10.e7))
"""

vollmer_time_df = pd.DataFrame(
	np.array((
		[10, 	-500], 
		[10, 	-590],
		[10, 	-670],
		[10, 	-600],
	)).T, 
	index = ['step', 'start'], 
	columns = ['4522', '4388', '4501', '4330']
)

vollmer_p_ram_df=pd.DataFrame(
	np.array((
		(5 * 80**2, 		80),
		(0.4*1.25e5, 		100),
		(4.*1.28e5, 		160),
		(2.0*0.25*5.e5, 	100)
	)).T, 
	index = ['peak', 't_HW_sq'], 
	columns = ['4522', '4388', '4501', '4330']
)

def voll_time_array(galaxy, *, i=None):
	step, start = vollmer_time_df[galaxy]
	if i is None:
		return np.arange(*vollmer_ranges[galaxy])*step + start
	r = vollmer_ranges[galaxy]
	if not r[0] <= i < r[1]:
		raise IndexError(
			f'voll_time_array({galaxy}): range=({r}), '
			f'valid indices: [{r[0]}, {r[1]}]: passed i={i}')
	return i*step+start

def voll_rp_profile(galaxy, *, i=None):
	peak, t_HW = vollmer_p_ram_df[galaxy]
	return peak / (voll_time_array(galaxy,
				   i=i if i is None else i+vollmer_ranges[galaxy][0])
				   ** 2 + t_HW ** 2)

# t_HW_4522=nan
# t_HW_sq4522=pow(t_HW_4522, 2)

"""
vollmer_p_ram__mag__t_HW_sq = {
	'4522': (5000, 80**2), 
	'4388': (0.4*1.25e17, 5.9e8), 
	'4501': (4.*1.28e17, 6.7e8/10**6),
	'4330': (2.0*0.25*5.e17, 6.e8/10**6)
}

vollmer_t_ram = {
	'4522': (10, -500), 
	'4388': (10e6, -590e6), 
	'4501': (10, -670),
	'4330': (10, -600)
}

def p_ram_raw(t, mag, t_HW_sq, off = 0): return mag * t_HW_sq / (t_HW_sq + (t - off) ** 2)
def p_ram_diff_raw(t, mag, t_HW_sq): return -mag * t_HW_sq * 2 * t / (t_HW_sq + (t ** 2)) ** 2
def p_ram(t, galaxy): return p_ram_raw(t, *vollmer_p_ram__mag__t_HW_sq[galaxy])
def p_ram_diff(t, galaxy): return p_ram_diff_raw(t, *vollmer_p_ram__mag__t_HW_sq[galaxy])

def t_ram_raw(index, mult, offset): return mult*index+offset
def t_ram_inverse_raw(t, mult, offset): return (t-offset)/mult
def t_ram(index, galaxy): return t_ram_raw(index, *vollmer_t_ram[galaxy])
def t_ram_inverse(galaxy, t): return t_ram_inverse_raw(t, *vollmer_t_ram[galaxy])
def t_ram_base(galaxy): return t_ram(np.arange(*vollmer_ranges[galaxy]), galaxy)

def voll_rp_profile(galaxy): return p_ram(t_ram_base(galaxy), galaxy)
def voll_rp_profile_diff(galaxy): return p_ram_diff(t_ram_base(galaxy), galaxy)"""

"""
def add_indexer_axis(host_ax, galaxy, ax2_offset=-.1+.025, label_offset=-.1, label_y=None):
	ax2=host_ax.twiny()
	ax2.set_frame_on(True); ax2.patch.set_visible(False)
	for sp in list(ax2.spines.values()): sp.set_visible(False)
	ax2.spines["top"].set_visible(True)
	ax2.spines["top"].set_position(("axes", ax2_offset))
	ax2.set_xlim(host_ax.get_xlim())
	#ax2.xaxis.set_major_locator(plt.MaxNLocator(len(host_ax.get_xticks())))
	ax2.set_xticklabels([int(t) for t in t_ram_inverse(galaxy, np.array(host_ax.get_xticks()))])
	if not label_y: label_y=(label_offset, ax2_offset+label_offset)
	for a, y, l in zip((host_ax, ax2), label_y, ('time', 'index')):
		a.xaxis.set_label_coords(0, y)
		a.set_xlabel(l, fontsize=14, ha='right')
	return ax2
"""

default_rp_plot_kw = dict(ls=':', lw=2, c='g')
default_rp_fill_kw = dict(color=color_darken('lightgray',.7), alpha=.55)
def add_rp_profile(#marker='+', markersize=10, additional_axes_to_legend=[],
					host_ax=None, host_fig=None, fontsize=None, legend=False, 
					rp_ax=True, X_array=None, by_index=False, legend_kw={}, 
					adjust_subplots=.85, RPax_position=1.1, plot_kwargs=None,
					fill=True, fill_kwargs=None,
					plot_by_1000=False, RP_ylabel=True, diff=False, galaxy=None, **kwargs):

	legend_kwargs=dict(fontsize=14, bbox_to_anchor=(.35, .885))
	legend_kwargs.update(legend_kw)
	#legend_kw=None
	
	if host_fig is None: host_fig=plt.gcf()
	if host_ax is None: host_ax=plt.gca()
	ax3=host_ax.twinx() if rp_ax else host_ax#create_hidden_axis(host_ax, label='RP_ax')#
	#ax3.axis('on')
	#for spk in ax3.spines.keys():
	#	if spk!='right': ax3.spines[spk].set_visible(False)
	
	if not fontsize: fontsize=14

	if X_array is None:
		X_array=np.arange(*vollmer_ranges[galaxy]) if by_index else voll_time_array(galaxy)#t_ram_base(galaxy)

	if diff:
		raise NotImplementedError('`diff` kwarg not currently supported for ram pressure profiles')
		#p_ram_plot=voll_rp_profile_diff(galaxy)/1000 if plot_by_1000 else voll_rp_profile_diff(galaxy)
	else:
		p_ram_plot=voll_rp_profile(galaxy)
		if plot_by_1000: p_ram_plot/=1000

	if fill:
		if fill_kwargs is None: fill_kwargs = {}
		fill_kwargs.update(kwargs)
		RPplot = ax3.fill_between(X_array, np.zeros_like(X_array), p_ram_plot,
								  label='Ram Pressure (max=5000)', **default_rp_fill_kw, **fill_kwargs)
		#print(f'add_rp_profile: fill:\nx:{X_array}\ny:{p_ram_plot}')
	else:
		if plot_kwargs is None: plot_kwargs = {}
		RPplot = ax3.plot(X_array, p_ram_plot,
						  label='Ram Pressure (max=5000)', **default_rp_plot_kw, **plot_kwargs)
		
	if RP_ylabel:
		#RPlabel=(RP_label_1000 if plot_by_1000 else RP_label)
		if rp_ax: ax3.set_ylabel(RP_label, size=fontsize, rotation=270, va='bottom')
		else: ax3.set_ylabel(RP_label, size=fontsize)

	if fill and not diff:
		ax3.set_ylim(ymin=0)
	
	if legend:
		print('legend in `add_rp_profile` not implemented')
		"""if fill: host_ax.fill_between()
		else: host_ax.plot(.5, .5, transform=host_ax.transAxes, label='Ram Pressure (max=5000)', **plot_kwargs)
		#L=host_fig.legend()
		#Lprop=host_fig.legend()
		handles, labels=host_ax.get_legend_handles_labels()
		for add_ax in additional_axes_to_legend:
			add_handles, add_labels=add_ax.get_legend_handles_labels()
			handles+=add_handles; labels+=add_labels
		L=host_ax.legend(handles, labels, **legend_kwargs)
		#L.remove()
		return ax3, L"""
	if rp_ax:
		if fill:
			ax3.set_zorder(-1)
			host_ax.patch.set_alpha(0)
		ax3.set_xlim(host_ax.get_xlim())
		pos=host_ax.get_position()
		ax3.spines["right"].set_position(("axes", RPax_position))
		
		for ax in (host_ax, ax3): ax.set_position(pos)
		
		if adjust_subplots:
			plt.subplots_adjust(right=adjust_subplots)

	#else:
	#	#ax3.spines["left"].set_position(("axes", -.1))
	#	return RPplot#ax3

	return ax3, RPplot
		
__all__ = ('vollmer_p_ram_df', 'voll_time_array', 'add_rp_profile', 'voll_rp_profile')

#RP_label=r'RP (g cm${\rm ^3\cdot}$km$\cdot$s${\rm ^{-1}}$)'
RP_label_1000=r'RP (10${\rm ^3\ g\ cm^{-3}\cdot\ km^2}\cdot$s${\rm ^{-2}}$)'
RP_label=RP_label_1000
RP_diff_char=r'$\dot{{\rmRP}}$'
RP_label_diff=RP_diff_char+RP_label[2:]
RP_label_1000_diff=RP_diff_char+RP_label_1000[2:]

if __name__=='__main__': #https://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
	pass
	# ax=plt.subplot(111)
	# ax.plot(t_ram_base('4522'), voll_rp_profile('4522'), marker='o')
	# ax2=add_indexer_axis(ax, '4522')
	# plt.show()
	