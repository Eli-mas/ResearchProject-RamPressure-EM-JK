import numpy as np, pandas as pd
from plot.plotting_functions import *
from prop.asy_prop import *
from prop.asy_prop_plot import *
from prop.asy_defaults import *
from comp.computation_functions import *
from comp.polar_functions import *
from comp.numba_funcs import get_tail_spectrum
from comp.array_functions import reindex
from plot.voll_rp_profile import add_rp_profile

from matplotlib import cm, pyplot as plt
# [(]((?!def)(.|\n))*?([)][:]).*

def figsave(self,title,append=None,close=True,**kw):
	title=self.title+' '+title
	if append is not None: title+=(' '+append)
	paper_tex_save(title,paper=2,close=close,**kw)

def miscsave(self,*a,**kw):
	self.figsave(*a,link_folder='--misc',close=True,**kw)

def namesave(self,title,append=None,**kw):
	self.figsave(title,append=append,link_folder=title,**kw)

def sigtimeplot(self,*a,**kw): return self.timeplot(*a,start=self.ind_sig,**kw)
def atimeplot(self,*a,**kw): return self.timeplot(*a,aplot=True,**kw)

@makeax
def ylim(self,center=None,ax=None):
	if center is None: center=self.WA
	ax.set_ylim(center+self.aplot_band)

def rplot(self):
	self.ratio_plot()
	self.ratio_plot(diff=True)
	self.ratio_plot(weights=True,which={0,})
	self.ratio_plot(weights=True,which={1,})

def aplot_weights(self,ax=None,**kw):
	self.aplot(ax=ax,which={0,},weights=True,**kw)
	self.aplot(ax=ax,which={1,},weights=True,**kw)

def weighted_ratio_plots(self,ax=None):
	for i,attr in zip((0,1),('EA','FA')):
		self.aplot(
			ax=ax,which=i,weights=True,
			save_args=(self.title+' '+attr+' weights aplot',),
			clf=True
		)

@paperplot2
@makeax
def shortsum_ER_plot(self,ax=None):
	ax2=ax.twinx()
	self.timeplot('ER',ax=ax)
	self.timeplot('shortsum',ax=ax2,c=mpl_colors[1])
	ax2.set_ylim(-.05,1.05)
	axlabels(LABELS.ER,'shortsum')
	full_legend(ax,ax2)

@paperplot2
@makeax
def shortsum_EA_plot(self,ax=None):
	ax2=ax.twinx()
	self.timeplot('EA',ax=ax,aplot=True,center=self.WA)
	self.timeplot('shortsum',ax=ax2,c=mpl_colors[1])
	ax2.set_ylim(-.05,1.05)
	hline(self.WA,ax=ax,label='wind')
	full_legend(ax,ax2,fontsize=16)
	axlabels(LABELS.EA,'shortsum')
	plt.suptitle(self.title+': shortsum, '+LABELS.EA)
	test_fig_save(self.title+' - shortsum, extent angle','shortside longside plots')
	plt.clf()

#@paperplot2
#@makeax
def tail_spectrum_plot(self,cutoffs=(.3,.5,.7,.85,.99)):#,ax=None
	results=self.apply(get_tail_spectrum,cutoffs,return_angles=True)
	"""
	AXES
		0 frame
		1 cutoff
		2 valtype (minspan,maxspan,mincenter,maxcenter)
	"""
	for i,q in zip((0,2),('spans','centers')):
		aplot=(i>1)
		ax=plt.subplot()
		if aplot:
			self.timeplot(results[:,:,i],aplot=True,center=self.antiWA)
			hline(self.antiWA)
			try: ax.set_ylim(self.antiWA+2*self.aplot_band)
			except AttributeError: pass
		else: self.timeplot(results[:,:,i])
		for l,c in zip(ax.lines[:],cutoffs): l.set_label('threshold=%g%%'%(100*c))
		ax.legend()
		if self.directory[0]=='v': add_rp_profile(host_ax=ax, galaxy=self.galaxy)
		#self.figsave('tail spectrum '+q,link_folder='tail spectrum')
		self.figsave(f'tail spectrum {q}', link_folder = ['tail spectrum', q])


@paperplot2
@makeax
def score_evolve_plot(self,array,angle,step=3,ax=None,cmap=cm.get_cmap("jet")):
	print('array:',array)
	for i,g in enumerate(self.instances[::step]):
		data=getattr(g,array)
		theta=(getattr(g,angle)+180)%360
		ratio=np.max(data[:,1])
		color=cmap(step*i/self.count)
		if 'flux' not in array: data=data*[rad_to_deg,1]
		ax.plot(*data.T,c=color,alpha=.5,zorder=-1)
		ax.scatter(theta,ratio,c=np.atleast_2d(color),zorder=1)
	scatter=np.array([c.get_offsets()[0] for c in ax.collections])
	colors=cmap(np.linspace(0,1,scatter.shape[0]-1))
	for i,z in enumerate(zip(scatter[:-1],scatter[1:])):
		plt.plot(*z.T,c=colors[i])
	vline((self.WA,self.WA+180),mod=360)

@makeax
def m1_m2_ER_plot(self,ax=None,lplot=True,lkw={},**kw):
	ax.scatter(self.m2ER,self.ER,c=self.time_index,**kw)
	ax.set_xlabel(LABELS.m2ER)
	ax.set_ylabel(LABELS.ER)
	if lplot or lkw:
		lkw={**dict(c='k'),**lkw}
		ax.plot(self.m2ER,self.ER,**lkw)

@makeax
def m1_m2_FR_plot(self,ax=None,lplot=True,lkw={},**kw):
	ax.scatter(self.m2FR,self.FR,c=self.time_index,**kw)
	ax.set_xlabel(LABELS.m2FR)
	ax.set_ylabel(LABELS.FR)
	if lplot or lkw:
		lkw={**dict(c='k'),**lkw}
		ax.plot(self.m2FR,self.FR,**lkw)

def flux_track_plot(self,ax=None,diff=False):
	ax=plt.subplot()
	
	for attr in ('noncenterflux','centerflux','totalflux'):
		self.timeplot(attr,ax=ax,diff=diff)
	ax2=ax.twinx()
	self.timeplot('FR',ax=ax2,diff=diff,ls='--' if diff else None)
	full_legend(ax,ax2)
	if diff:
		hline(0,ax=ax,zorder=-1,c='k')
		hline(0,ax=ax2,zorder=-1,c='purple',ls=':',lw=2)
	if self.directory[0]=='v': add_rp_profile(host_ax=ax,diff=diff, galaxy=self.galaxy)
	title=self.title+' flux track plot'
	if diff: title+=' diff'
	plt.suptitle(title)
	paper_tex_save(title,paper=2,link_folder='flux track')

def _ext_tack_calc(self,mode,window,roll):
	if window == 180: reftype = 'E'
	elif window == 90: reftype = 'qE'
	halfwindow = window / 2
	window_ind = int(window * deg_to_index)
	ref = getattr(self, reftype + 'A')

	if mode=='wind':
		short_center=self.WA
		reindexer=reindex
	elif mode=='find':
		short_center=polar_reduction(np.median,ref,mod=360)
		reindexer=reindex
	elif mode=='short':
		short_center=ref
		reindexer=complex_reindex_2d
		#shorts=self.shortside_list_graph[:,:,1]
		#shorts[np.abs(polar_offset(self.shortside_list_graph[:,:,0],np.full([self.count,1],self.WA)))>90]=nan
	try: short_start_ind=int((short_center-halfwindow)*deg_to_index)
	except TypeError: short_start_ind=((short_center-halfwindow)*deg_to_index).astype(int)
	shorts=reindexer(self.extentlist_graph[:,:,1],short_start_ind,axis=1)[:,:window_ind]

	"""if (mode=='wind') or (mode=='find'):
		longs=reindex(self.extentlist_graph[:,:,1],short_start_ind+al,axis=1)[:,:al]
	elif mode=='short':
		longs=self.longside_list_graph[:,:,1]
		longs[np.isnan(shorts)]=nan
		#lprint(self.title,longs[0])"""
	longs = reindexer(self.extentlist_graph[:, :, 1], short_start_ind + al, axis=1)[:, :window_ind]
	return shorts, longs, reftype

@makeax
def ext_track_plotter(self, roll, mode, window, ax=None):
	shorts, longs, reftype = self._ext_tack_calc(mode, window, roll)

	if roll:
		ax.plot(self.time_index[1:], np.nanmean(shorts[1:] / shorts[:-1], axis=1), label='short')
	elif mode == 'short':
		self.timeplot(reftype + 'R', ax=ax)
	# self.sig_angle_plot('EA')
	if roll: hline(1, zorder=-1, ls='-', label='wrt rolling')

	ax2 = ax.twinx()
	if mode == 'short':
		shortplot = np.mean(shorts, axis=1)
		shortplot /= shortplot[0]
	else:
		shortplot = np.nanmean(shorts / shorts[0], axis=1)
	self.timeplot(shortplot, c=mpl_colors[0], ls='--', label='' if roll else 'short', ax=ax2)
	l, = hline(1, zorder=-1)
	if roll: l.set_label('wrt original')

	if roll:
		ax.plot(self.time_index[1:], np.nanmean(longs[1:] / longs[:-1], axis=1), c=mpl_colors[1], label='long')
	if mode == 'short':
		longplot = np.mean(longs, axis=1)
		longplot /= longplot[0]
	else:
		longplot = np.nanmean(longs / longs[0], axis=1)
	self.timeplot(longplot, c=mpl_colors[1], ls='--', label='' if roll else 'long', ax=ax2)
	return ax, ax2

def ext_track_plot(self,roll=False,mode='wind',window=180):
	ax, ax2=self.ext_track_plot(roll, mode, window)

	if mode!='short': ax.set_yticks([])
	full_legend(ax,ax2,ncol=2)
	
	if self.directory[0]=='v': add_rp_profile(host_ax=ax, galaxy=self.galaxy)
	
	title='ext track'
	if roll: title+=' roll'
	title+=' mode='+mode+' window=%s'%window
	plt.suptitle(title)
	self.figsave(title,link_folder=('ext track',window))


def track_extent_sectors(self):
	"""
	former comments/code in 4-19-19 non-decommented version
	
	the code executed in this function better shows why qEA outperforms EA for R0
	it may come in handy elsewhere
	"""
	
	
	extentlist=self.extentlist_graph[:,:,1]
	for half,attr in zip((0,aql,ahl),('raw','qEA','EA')):
		if half: extents=rolling_gmean_wrap(extentlist,half,axis=1)
		else: extents=extentlist
		
		short_locs=np.argmin(extents,axis=1)
		long_locs=np.argmax(extents,axis=1)
		
		try: self.timeplot(attr,aplot=True)
		except:
			self.timeplot('EA',aplot=True,zorder=-1)
			self.timeplot('qEA',aplot=True,zorder=-1)
		
		for ar,c in zip((short_locs,long_locs+al), mpl_colors[:2]):
			self.timeplot(ar*index_to_deg,aplot=True,c=c,ls=':',lw=2)
		
		hline(self.WA,zorder=-1)
		
		plt.suptitle(self.directory+' '+attr)
		#self.figsave('aplot components '+attr,link_folder='aplot components')
		self.namesave('aplot components',append=attr)

def angular_property_sector_track(self,
	property,center_angle,span=180,
	axes=None,axgroup=None,baseaxes=None,
	adjust=0,
	frame_drag=False,frame_step=3,frame_count=5,
	indices=None,fmt=True,
	**kw
):
	"""
	`property` can be
		a string, in which case it can be interpreted in two ways:
			(1) as the attribute that will be passed to the `Galaxy.plot_angular_property` function
				for plotting for each Galaxy instance
			(2) as the function that will be called directly for each Galaxy instance,
				which should follow the argument signature for `Galaxy.plot_angular_property`
		
		a function, in which case
			it should accept a Galaxy instance as its first argument
				(all other arguments (positional or keyword) will be passed in as keywords)
			it should conduct plotting internally, and therefore accept an 'ax' keyowrd argument
		
		an array, in which case
			if it is a 2-d array, AXES={0 frame, 1 theta}, data=(value)
			if a 3-d array, AXES={0 frame, 1 theta, 2 qtype (angle | value)}, data=(angle,value)
				
	"""
	#init=True
	
	if baseaxes is not None:
		axes=np.array([ax.twinx() for ax in baseaxes])
	elif axes is None:
		fig,axes=plt.subplots(2,3,sharex=True)#,sharey=True
		axes=np.ravel(axes)
		#init=False
	else: axes=np.atleast_1d(axes)
	
	if indices is None: indices=self.sig_ind_linspace
	else: indices=np.atleast_1d(indices)
	
	for ind, (ax,t,index) in enumerate(zip(axes,self.time_index[indices],indices)):
		for i,frame in enumerate(filter(
			lambda v: v>=0,
			range(index-frame_step*frame_count+frame_step,index+frame_step,frame_step)
		)):
			if (not frame_drag) and (frame<index): continue
			#print('ylim before plot:',ax.get_ylim())
			
			alpha=(i+1)/frame_count
			if isinstance(property,str):
				Ginstance=self.instances[frame]
				attr=getattr(Ginstance,property)
				if callable(attr):
					attr(center_angle, span, ax=ax, adjust=adjust, alpha=alpha, **kw)
				else:
					Ginstance.plot_angular_property(
						attr, center_angle, span,
						ax=ax, adjust=adjust,
						alpha=(i+1)/5, **kw
					)
				format=True
			elif callable(property):
				raise NotImplementedError()
				format=False
			#elif isinstance(property,ndarray):
			else:
				self.instances[frame].plot_angular_property(
					property[index], center_angle, span,
					ax=ax, adjust=adjust, alpha=alpha,
					**kw
				)
				format=True
			
			if (frame==index) and format:
				xt=center_angle+adjust+np.linspace(-span/2,span/2,5)#np.arange(-span/2,span/2+30,30)
				ax.set_xticks(xt)
				ax.set_xticklabels((xt%360).round(1))
				ax.text(.99,.99,'t=%i'%t,va='top',ha='right',transform=ax.transAxes,fontsize=20)
			#print('ylim after plot:',ax.get_ylim())
	
	if format and fmt:
		if axgroup is None: axgroup=(axes,)
		for group in axgroup:
			if isinstance (group[0],int): ref=axes[group]
			else: ref=group
			make_common_axlims(*ref)
			for ax in ref: vline(center_angle+adjust,mod=360,ax=ax)
	
	return axes

def plot_property_by_windside(self,mode,**kw):
	content='frame_drag' if kw.get('frame_drag') else 'singular'
	folder='windside track/%s/%s/'%(content,mode)
	if mode=='ext': func=self.boundary_sector_track
	elif mode=='flux_total': func=self.total_flux_sector_track
	elif mode=='flux_outer': func=self.outer_flux_sector_track
	
	file_append=' '+mode+' '+content
	axes=func(self.WA,c=mpl_colors[0],**kw)
	self.figsave('windside track'+file_append,link_folder=folder+'windside',clf=False,close=False)
	
	newaxes=tuple(ax.twinx() for ax in axes)
	func(self.WA+180,c=mpl_colors[1],adjust=180,axes=newaxes,**kw)
	self.figsave('windside-leeside track'+file_append,link_folder=folder+'both',clf=False,close=False)
	
	xticks=np.linspace(self.WA-90,self.WA+90,5)
	for ax,ax2 in zip(axes,newaxes):
		ax.remove()
		ax2.set_xticks(xticks)
		ax2.set_xticklabels((xticks+180)%360)
	self.figsave('leeside track'+file_append,link_folder=folder+'leeside')

def sig_angle_plot(self,*angles,ref=None,threshold=20,mod=360,ax=None,**kw):
	cond=np.full(self.count,True)
	if ref is None: ref=self.WA
	for angle in angles:
		if isinstance(angle,str): theta=getattr(self,angle)
		else: theta=angle
		cond*=(np.abs(polar_offset(theta,ref,mod=mod))<=threshold)
	
	sig_region_plot(cond,X_array=self.time_index,ax=ax)

def flux_track(self):
	for diff in (False,True): self.flux_track_plot(diff=diff)

def ext_track(self,**kw):
	#for roll in (False,True): self.ext_track_plot(roll=roll,**kw)
	for mode,window in product(('short','wind','find'),(90,180)):
		self.ext_track_plot(mode=mode,window=window,**kw)

def boundary_sector_track(self,*a,**kw):
	return self.angular_property_sector_track('plot_outer_boundary_sector',*a,**kw)

def total_flux_sector_track(self,*a,**kw):
	return self.angular_property_sector_track('plot_total_flux_sector',*a,**kw)

def outer_flux_sector_track(self,*a,**kw):
	return self.angular_property_sector_track('plot_outer_flux_sector',*a,**kw)

def windside_property_plots(self):
	for mode in ('ext','flux_total','flux_outer'):
		self.plot_property_by_windside(mode)
		self.plot_property_by_windside(mode,frame_drag=True)

def extent_ratio_arrays_track(self):
	for attr in ('extentratio','score','qscore','score_trig','qscore_trig'):
		self.extent_ratio_array_track_plot(attr)

@makeax
def ratio_array_fill(self,angles,ratios,ax=None):
	pos=ratios>=1
	for v,c in zip((1,np.mean(ratios[pos])),mpl_colors[1:]):
		if v==1: cond=pos
		else:
			continue
			cond=ratios>=v
		for ar in get_region_inds(cond):
			ax.fill_between(
				stabilize_angles(angles[ar],360),
				np.ones_like(ar),ratios[ar],
				color=c,alpha=.5)
		#print(stabilize_angles(angles[cond],360)[[0,-1]])
	ax.set_xlim(0,360)
	hline(1,ax=ax)
	vline(self.antiWA,ls='-',ax=ax,lw=2)
	vline(angles[np.argmax(ratios)],c='r',ax=ax,lw=3)
	vline(polar_reduction(np.median,angles[pos],360),c='blue',ax=ax,ls=':',lw=3)

def extent_ratio_array_track_plot(self,attr):
	fig,axes=plt.subplots(2,3,gridspec_kw={'hspace':.1,'wspace':.1})#sharex=True,
	axes=np.ravel(axes)
	indices=np.s_[self.sig_ind_linspace]
	
	for index, ax, t in zip(indices, axes, self.time_index[indices]):
		angles,ratios=getattr(self,attr)[index,:,:].T
		angles*=rad_to_deg
		ax.plot(angles,ratios)
		self.ratio_array_fill(angles,ratios,ax=ax)
		#ax.set_xticks((center_angle+adjust+np.linspace(-90,90,5))%360)
		ax.text(.01,.99,'t=%i'%t,va='top',ha='left',transform=ax.transAxes,fontsize=20)
	
	for ax in axes[:3]:
		ax.yaxis.tick_right()
		#ax.tick_params(direction='in', axis='x')
		#for ticklabel in ax.get_xticklabels(): ticklabel.set_visible(True)
	
	self.figsave(attr+' structure track',link_folder='ratio structure/'+attr)

@makeax
def extent_arrays_grouped_plot(self,index,ax=None):
	for attr,c in zip(('extentratio','qscore','score'),('g',qext_color,ext_color)):
		angles,ratios=getattr(self,attr)[index,:,:].T
		angles*=rad_to_deg
		ax.plot(angles,ratios**(1/np.max(np.log(ratios))),c=c)
		if 'score' in attr: vline(angles[np.argmax(ratios)],c=c,ax=ax,lw=2)
	
	ax.set_yticks([1])
	ax.set_xlim(0,360)
	hline(1,ax=ax)
	vline(self.antiWA,ls='-',ax=ax,lw=2,zorder=-1)

def grouped_extents_track(self):
	fig,axes=plt.subplots(2,3,gridspec_kw={'hspace':.1,'wspace':.1})#sharex=True,
	axes=np.ravel(axes)
	indices=np.s_[self.sig_ind_linspace]
	for index, ax, t in zip(indices, axes, self.time_index[indices]):
		self.extent_arrays_grouped_plot(index,ax=ax)
		ax.text(.01,.99,'t=%i'%t,va='top',ha='left',transform=ax.transAxes,fontsize=20)
	self.figsave('grouped extents track',link_folder='grouped extent arrays track')





__all__ = ('figsave','miscsave','namesave','sigtimeplot','atimeplot','ylim','rplot',
'aplot_weights','weighted_ratio_plots','shortsum_ER_plot','shortsum_EA_plot',
'tail_spectrum_plot','score_evolve_plot','m1_m2_ER_plot','m1_m2_FR_plot',
'flux_track_plot','_ext_tack_calc','ext_track_plotter','ext_track_plot',
'track_extent_sectors','angular_property_sector_track','plot_property_by_windside',
'sig_angle_plot','flux_track','ext_track','boundary_sector_track',
'total_flux_sector_track','outer_flux_sector_track','windside_property_plots',
'extent_ratio_arrays_track','ratio_array_fill','extent_ratio_array_track_plot',
'extent_arrays_grouped_plot','grouped_extents_track',)
