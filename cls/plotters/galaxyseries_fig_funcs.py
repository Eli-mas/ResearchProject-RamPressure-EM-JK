import numpy as np
from matplotlib import gridspec, pyplot as plt
from plot.plotting_functions import *
from prop.asy_prop import *
from prop.asy_prop_plot import *
from prop.asy_defaults import *
from comp.computation_functions import *

def rp_trace_plot(self, reject_by_m2 = True, save = True,
				  sig_asy = True, plot_sig_asy = True, zoom=True):
# 	PA_radii=self.get_PA_radii()#np.sum(PA_radii, axis=1)
# 	gas_content=self.get_contained_gas()
	
	ax_count=6
	gs=gridspec.GridSpec(ax_count, 6, hspace=0, wspace=0)
	#fig, axes=plt.subplots(3, 1, gridspec_kw={'hspace':0}, sharex=True)
	axes=tuple(plt.subplot(gs[i, :]) for i in range(1, ax_count))
	gas_ax, ratio_ax, m1_m2_ext_ax, angle_ax, angle_zoom_ax=axes
	radius_ax=gas_ax.twinx()
	gas_ax.plot(self.time_index, self.gas_content, c='b', label=r'Gas Mass (within initial region) [M$\odot$]')
	radius_ax.plot(self.time_index, self.PA_diameters, c='r', label='Gas Diameter (major axis) [kpc]')
	gas_ax.set_ylabel('gas mass')
	radius_ax.set_ylabel('gas diameter', rotation=270, va='bottom')

	full_legend(radius_ax, gas_ax, fontsize = 20)

	norm0_ratio_ax = self.ratio_plot(ax=ratio_ax, save=False, m2=False)#, show_rp=False
	ratio_ax.get_legend().set_title('asymmetry measures', prop=dict(size=20))
	if sig_asy:
		try:
			if not reject_by_m2: raise AttributeError
			sig_asy=sig_asy_full_mask(self.ER, self.FR, self.m2ER, self.m2FR)
			print('sig_asy in rp_trace_plot: m=2 included')
		except AttributeError:
			print('AttributeError in sig_asy in rp_trace_plot, no m=2 included')
			sig_asy = sig_asy_mask(self.ER, self.FR)
		self.aplot(ax=angle_ax, save=False, sig_asy=sig_asy, plot_sig_asy=plot_sig_asy, show_rp=True, offsets=True)
		self.aplot(ax=angle_zoom_ax, save=False, sig_asy=sig_asy, zoom=True, plot_sig_asy=plot_sig_asy, show_rp=False, offsets=True)
	else:
		sig_asy = np.full(len(self),True,dtype=bool)
		self.aplot(ax=angle_ax, save=False, sig_asy=sig_asy, plot_sig_asy=plot_sig_asy, show_rp=True, offsets=True)
		self.aplot(ax=angle_zoom_ax, save=False, sig_asy=sig_asy, zoom=False, plot_sig_asy=plot_sig_asy, show_rp=False, offsets=True)
		print(angle_zoom_ax)
		sig_asy = None
	#print(sig_asy)
	angle_ax.get_legend().set_title('asymmetry angles', prop=dict(size=20))
	angle_zoom_ax.legend(handles=[], title='asymmetry angles (zoomed)', title_fontsize=20)

	#m1_m2_flux_ax=m1_m2_ext_ax.twinx()
	if self.inc <= HIGH_I_THRESHOLD:
		m1_m2_flux_ax=self.ratio_plot(('ER','FR'), ax=m1_m2_ext_ax, title=False, save=False, m2=True, show_rp=False)
		for s, m, l in zip((None, np.s_[::2]), (None, 'o'), (None, 'ER/m2ER')):
			self.timeplot(
				self.ER/self.m2ER, c=ext_color, marker=m, 
				ax=m1_m2_ext_ax, label=l, slice=s)#'lightblue'
		for s, m, l in zip((None, np.s_[::2]), (None, 'o'), (None, 'FR-m2FR')):
			self.timeplot(
				self.FR-self.m2FR, c=flux_color, marker=m, 
				ax=m1_m2_flux_ax, label=l, slice=s)#color_lighten('purple')
		m1_m2_ext_ax.legend().remove()
		full_legend(m1_m2_flux_ax, m1_m2_ext_ax, ncol=2).set_title('m1 | m2', prop={'size':20})
		#m1_m2_flux_ax.set_xticks([])
		m1_m2_flux_ax.set_ylabel('0-norm ratios', rotation=270, va='bottom')
	else:
		pass
		#add_rp_profile(host_ax = m1_m2_ext_ax, galaxy=self.galaxy)

	for ax in (m1_m2_ext_ax, ratio_ax): ax.set_ylabel('1-norm ratios')
	norm0_ratio_ax.set_ylabel('0-norm ratios', rotation=270, va='bottom')


	#paper_tex_save(self.title+' rp trace plot', paper=2, clf=True)
	#self.figsave('rp trace plot', link_folder='rp trace')
	for ax in axes[:-1]: ax.set_xticks([])
	#angle_zoom_ax.legend().remove()

	
	if zoom:
		lines = angle_zoom_ax.lines[:]
		angle_zoom_ax_yminima = [np.nanmin(l.get_data()[1]) for l in lines]
		angle_zoom_ax_ymin = max(angle_zoom_ax_yminima)
		angle_zoom_ax_ymaxima = [np.nanmax(l.get_data()[1]) for l in lines]
		angle_zoom_ax_ymax = max(angle_zoom_ax_ymaxima)

		removals = []
		# print(f'rp_trace_plot {self.directory}:')
		for i, (l, m) in enumerate(zip(lines, angle_zoom_ax_yminima)):
			# print(np.nanmin(l.get_data()[1]), np.nanmax(l.get_data()[1]))
			if angle_zoom_ax_ymax-m > 360:
				# print('remove')
				removals.append(i)
				continue

		for i in removals: lines[i].remove()
	else:
		for l in angle_zoom_ax.lines[:]:
			m,M = l.get_data()[1].min(), l.get_data()[1].max()
			if m<angle_ax.get_ylim()[0] or M> angle_ax.get_ylim()[1]:
				l.remove()
	
	imshow_axes=np.array([plt.subplot(gs[0, i]) for i in range(6)])
	try:
		self.zshow(
			axes=imshow_axes, save=False, wind=True, 
			indices=[self.tindex(t) for t in self.rp_trace_zshow_indices]
		)
	except AttributeError:
		self.zshow(axes=imshow_axes, save=False, wind=True)

	imshow_axes[0].text(0.01, 0.01, 'a', va='bottom', ha='left', size=16, transform=imshow_axes[0].transAxes)
	imshow_axes[0].text(0.02, 1.03, self.title.upper(), va='bottom', ha='left', size=18, transform=imshow_axes[0].transAxes)

	for ax, c in zip(axes, ('b', 'c', 'd', 'e', 'f')):
		ax.text(0.01, 0.01, c, va='bottom', ha='left', size=16, transform=ax.transAxes)

	multi_axis_labels(axes, x='Time (Myr)', size=16)
	if save: self.namesave('rp trace', append='plot', s=(13, 16))
	
	return imshow_axes, axes

def zshow(self, indices=None, axes=None, save=True, save_kw={}, wind=False, inout=False, border=True):
	save=save and (indices is None) and (axes is None)
	if axes is None:
		fig, axes=plt.subplots(2, 3, gridspec_kw={'hspace':.1, 'wspace':.1})#sharex=True, 
		axes=np.ravel(axes)
	if indices is None:
		indices=np.s_[self.sig_ind_linspace]
		alt=False
	else:
		indices=np.round(indices).astype(int)
		alt=True
	zdata_nan=znan(self.zdata)
	vmin, vmax=np.nanmin(zdata_nan), np.nanmax(zdata_nan)
	del zdata_nan
	arrow_factors=[0 for i in indices]
	for j, (index, ax, t) in enumerate(zip(indices, axes, self.time_index[indices])):
		g=self.instances[index]
		g.zshow(ax=ax, vmin=vmin, vmax=vmax, border=border, inout=inout, axis=False)
		#ax.invert_yaxis()
		ax.text(.99, .99, 't=%i'%t, va='top', ha='right', transform=ax.transAxes, fontsize=20)
		axis=ax.axis()#keepax(ax=ax)
		WA_ind=int(self.WA*deg_to_index)
		offset=al//3
		"""
		box_radii=g.get_box_radii(trunc_y1=.2)*g.pix_scale_arcsec
		window_deg=60
		window=int(.5*window_deg*deg_to_index)
		averaged_extents=rolling_mean_wrap(box_radii-g.extentlist_graph[:, 1], window, 0)
		region=averaged_extents[(WA_ind+np.arange(offset, (-offset)%a2l+1))%a2l]
		loc=np.argmax(region)+WA_ind+offset
		outermost=np.max(reindex(g.extentlist_graph[:, 1], loc-aql)[:ahl])
		loc*=index_to_rad
		rot_radius=box_radii[int(loc)]
		print('rot_radius, outermost:', (rot_radius, outermost))
		frac=.35
		while (frac*rot_radius*self.instances[0].pix_scale_arcsec<outermost*1.1) and (frac<=1):
			frac+=.025
		#print(self.title+' t=%i loc (deg), outermost, frac:'%self.time_index[index], loc*rad_to_deg, outermost, frac)
		"""
		radius=np.min(np.abs(np.array([self.xpix, self.xpix, self.ypix, self.ypix])-ax.axis()))#(axis[1]-axis[0])/2
		if wind:
			WA=self.WA*tau/360
			polar_on_cart_plot(
				[WA, WA], [radius*.95, radius*.25], 
				xc=self.xpix, yc=self.ypix, 
				ax=ax, c='r', arrow=True
			)
		"""
		if self.rotation:
			#arrow_theta=np.linspace(loc-self.rotation*tau/8, loc+self.rotation*tau/8, 100)
			arrow_radius=rot_radius*(frac+.025)/g.pix_scale_arcsec
			arrow_boundary=reindex(self.extentlist[:, 1], -aql)[:ahl]
			arrow_std=np.std(arrow_boundary/np.max(arrow_boundary))
			#print('arrow_factors assignment', j)
			arrow_factors[j]=(j, loc, arrow_radius, arrow_std)
		"""
	if self.rotation:
		#print('arrow_factors:', arrow_factors)
		"""i, loc, r, sd=arrow_factors[0]#max(arrow_factors, key=lambda d: d[-1]*d[-2])"""
		i=0
		axis=axes[0].axis()
		xspan, yspan=axis[1]-axis[0], axis[3]-axis[2]
		try:
			xa, ya=rp_trace_rot_arrow_adjusts[self.title]
			xpos, ypos = axis[0] + xspan * xa, axis[2] + yspan * ya
		except KeyError:
			xpos = axis[1] - xspan/4
			ypos = axis[2] + yspan/4
		tcenter=c_to_p(xpos, ypos, self.xpix, self.ypix)[0]
		r=.25*np.average([xspan, yspan])
		t=np.linspace(tcenter-self.rotation*tau/8, tcenter+self.rotation*tau/8, 100)
		#print(self.title+' r, xc, yc, axis:', r, xpos, ypos, axis)
		#print('xa, ya, axis[0]:', xa, ya, axis[0])
		#print('xpos, ypos, self.xpix, self.ypix, tcenter:', xpos, ypos, self.xpix, self.ypix, tcenter)
		#print('c_to_p(xpos, ypos, self.xpix, self.ypix):', c_to_p(xpos, ypos, self.xpix, self.ypix))
		polar_on_cart_plot(t, np.full(100, r), xc=xpos+np.sign(self.xpix-xpos)*r/2, yc=ypos+np.sign(self.ypix-ypos)*r/2, ax=axes[i], c='b', arrow=True)
	"""
	try:
		z = np.array([np.where(self.instances[t].zdata) for t in indices])
		#print(z)
		## AXES 0 index, 1 x\y, 2 pixel
		(xmin, ymin), (xmax, ymax) = minmax(z, axis=(0, 2))
		axis = np.array([xmin, xmax, ymin, ymax])
		print(axis)
		del z
		for ax in axes: ax.axis(axis)
	except Exception as e:
		print(f'axis reframing could not be performed in zshow: <{repr(e)}>: see galaxyseries_fig_funcs.py')
	"""
	if not save: return
	if alt:
		#self.figsave('gas maps %s'%((indices[0], indices[-1], len(indices)), ), link_folder='gas maps')
		self.miscsave('gas maps', append='%s'%((indices[0], indices[-1], len(indices)), ), **save_kw)
	else:
		#self.figsave('gas maps', link_folder='gas maps')
		self.namesave('gas maps', **save_kw)




__all__ = ('rp_trace_plot', 'zshow')
