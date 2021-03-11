"""Underlying i/o routines, including:

	save_fmt: save a pandas DataFrame in human-readable format
."""

import sys, os, os.path, traceback, re, shutil, errno
from subprocess import run
from functools import partial, reduce

from common import consume

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS

from asy_io.asy_paths import *
from cls.classes.Galaxy.galaxy_attribute_information import \
	baseline_attributes, observational_only_attributes
from core import MultiIterator

from prop.galaxy_file import *
from prop.simprop import vollmer_center_coors, vollmer_psa, vollmer_PA
from prop.asy_prop import *
from prop.sim_fn_gen import sim_fn_x,sim_fn_y

# from cls.adc_prep import basic_info_values, real_data_values, noise_data_values

def load_array(path, **kw):
	return np.load(path, **kw)

def save_array(path, arr):#, tests=False, full_tests=True, ret_mmap=False
	"""save numpy file (*.npy) of numpy array and assert correct loading via memory map"""
	arr = np.atleast_1d(arr)
	np.save(makepath_from_file(path), arr)
	return arr

def save_fmt(savepath,*save_args,delimiter='\t',index_col=0,data=None,**read_kw):
	"""Save a pandas DataFrame with more readable formatting"""
	read_kw.update(dict(delimiter=delimiter,index_col=index_col))
	if data is None: data=pd.read_csv(savepath,**read_kw)
	
	with pd.option_context('display.max_columns',None,*save_args):
		dstr=data.to_string().split('\n')
	
	del data
	
	dstr[1]=dstr[1].replace(' ','')
	dstr[1]=dstr[1]+dstr[0][len(dstr[1]):]
	with open(savepath.replace('.file','_fmt.file'),'w+') as f:
		f.write('\n'.join(dstr[1:]))

def merge_dicts(*dicts,**k):
	d0={}
	consume(d0.update(d) for d in dicts)
	if k: d0.update(k)
	return d0

def d_beam_func(d1,d2): return np.sqrt(d1*d2)

def print_update(*p, **kw):
	"""Overwrite the contents of the current line, without creating a newline.
	
	The arguments passed are submitted to the inbuilt `print` function.
	Keywords may also be passed, except that 'flush' will always be
	set to True, and newline characters will be removed from 'end'.
	"""
	kw['flush'] = True
	kw.setdefault('end', '')
	kw['end'] = kw['end'].replace('\n','').replace('\r','')
	print('\x1b[2K\r'+' '.join([str(i) for i in p]), **kw)

def makepath(p): # see https://stackoverflow.com/questions/273192/
	"""
	Make a directory path, creating intermediary directories as required.
	:param p:
	:return:
	"""
	path = os.path.join(os.getcwd(), p)
	try:
		os.makedirs(path)
	except OSError:
		if not os.path.isdir(path):
			raise
	return path

def makepath_from_file(f):
	"""
	Make a directory to host a file given the full file path.
	
	The path is taken as everything before the final / character.
	:param f: full file path
	:return:
	"""
	makepath(f[:len(f)-f[::-1].index('/')])
	return f

def touch_directory(directory, *, errnum=errno.ENOENT, errmsg=os.strerror(errno.ENOENT)):
	"""
	Pass a directory and run the unix 'touch' command on it.
	"""
	if not os.path.exists(directory):
		raise FileNotFoundError(errnum, errmsg, directory)
	subprocess.run(['touch', directory])

def savetxt(p, ar, **k):
	"""Given a path to a file `p` and an array `ar`,
	call np.savetxt(p, ar, **kw), first ensuring that
	the directory containing p exists."""
	makepath_from_file(p)
	np.savetxt(p,ar,**k)

def str_replace(s,args,r=''):
	"""Given a replacement str `r`, for each str 'rep' in `args`,
	set s to be the result of s.replace(rep, r), and return the result.
	"""
	for arg in args: s=s.replace(arg,r)
	return s

def fpull(file, c=None, s=0, e=0, rep=None, r=None, ret=0):#, a=[]
	"""An OLD function that reads data from a file and stores
	in a list of lists. The lines are first split by str.split,
	with no keyword arguments passed. Then, the other parameters
	to this function modify how the data are handled.
	
	:: Parameters ::
	c: if provided, an integer; lines are only included
		if they have at least this many elements.
	
	s: starting line in the file to process; default 0 (first line)
	e: ending line (exclusive) in the file to process.
	
	rep: if provided, an iterable of str. Each element
		will be replaced by the value passed to the `r` parameter.
	r: the replacement str for each element in the `rep` argument.
		Has no effect if `rep` is not provided.
	
	ret: if bool(ret) is False, return dictionary
		where each line's first element comprise keys,
		with corresponding values being a list of remaining elements
		on the line.
	"""
	f=open(file,errors='replace')
	if not e: l=f.readlines()[s:]
	else: l=f.readlines()[s:e]
	if rep==None: l=[pl.split() for pl in l]
	else: l=[str_replace(pl,rep,r=r).split() for pl in l]
	if c==None: pass
	else: l=[pl for pl in l if len(pl)>c]
# 	if not a: pass
# 	else:
# 		for pl in l: pl[a[1]]=str_replace(pl[a[1]],a[0])
	f.close()
	if not ret:
		d={}
		for pl in l: d[pl[0].lower()]=pl
		return d
	elif ret=='a': return np.array(l)

# coordinate data for cluster galaxies
coorslinesdata=fpull(DATA_SOURCES_PATH+'gal_coor.file',rep=('h','m','s','d'),r=' ')
# other galaxy properties for cluster galaxies
shapedata=fpull(DATA_SOURCES_PATH+'gal_shape_mod.file')

OTHER_data=fpull(DATA_SOURCES_PATH+'OTHER/OTHER.file',s=1,e=2)
ATLAS3D_data=fpull(DATA_SOURCES_PATH+'ATLAS3D/ATLAS3D.file',s=1,e=24)

# also data for cluster galaxies, but used in other scripts
cluster_shape_data = pd.read_csv(DATA_SOURCES_PATH+'gal_shape_mod.data',index_col=0)
cluster_coor_data = pd.read_csv(DATA_SOURCES_PATH+'gal_coor.data',index_col=0)


def listify(input,convert): return [input] if isinstance(input,convert) else input
def listify2(input):
	try: return list(input)
	except TypeError: return [input]

def get_shortside_data(Ginstance, deny_m2_high_i=deny_m2_high_i):
	"""Get an array of shortside extents at each angle and their median.
	
	The return value is a tuple (e, m), where e is a 2-d array of the
	extents (second column) and the associated angles (first column),
	and m is the median of these extents. If the galaxy has low inclination,
	the extent values are deprojected, and the extents are divided by
	their median; otherwise, the extents are not deprojected, and the
	median is returned as nan.
	
	"""
	shortside_list_graph=np.array(Ginstance.shortside_list_graph)
	
	if deny_m2_high_i and Ginstance.inclination>high_i_threshold:
		shortside_show=shortside_list_graph
		shortside_median=nan
	else:
		shortside_show=np.array(Ginstance.shortside_list_graph_deproject)
		shortside_median=np.median(shortside_show[:,1])
		shortside_show[:,1]/=shortside_median
	
	shortside_show[:,0]=shortside_show[:,0]%360
	
	return shortside_show, shortside_median

def get_raw_data_from_fits(Ginstance):
	"""Given a Galaxy instance, extract relevant data from the fits file
	for that galaxy, set the instances 'data_raw' attribute, and return
	the fits file header (the file is closed first)."""
	filename,openname,filetype=Ginstance.filename,Ginstance.openname,Ginstance.filetype
	fits_file = fits.open(openname)
	data = fits_file[0].data
	# usually this is a 2-d array; if not,
	# assume that the data is contained along the last two axes
	while len(data.shape)>2: data=data[0]
	fileheader=fits_file[0].header
	fits_file.close()

	"""if Ginstance.is_rsim:
		#makepath(RESEARCH_DIRECTORY+'/Programming_Resources/Images/Outputs/Newer Versions/Other/New/Combined from asymmetry(.py)/sim/'+filename)
		PA=0
		sim_cutoff=sim_back_cutoff[filename[0]]
		#elif filename=='x202': sim_cutoff=.1
		
		data=np.copy(data)
		data[data<sim_cutoff]=0"""
	Ginstance.data_raw=np.array(data,dtype=float)
	
	return fileheader

def fits_data(Ginstance):
	"""Returns relevant data pertaining to galaxy and corresponding fits file.
	
	In particular, the following attributes are set on Ginstance:
		PA: position angle of the galaxy's major axis
		inclination: viewing angle (inclination) in degrees; 0 <= i <= 90
		xpix: x-coordinate of center as a pixel location on moment-0 map
		ypix: y-coordinate of center as a pixel location on moment-0 map
		pix_scale_arcsec: scale of side length of each pixel in arcsec
		d_beam_arcsec: beam diameter of the moment-0 map in arcsec
		is_atlas3d: boolean, tells if galaxy is in the ATLAS3D set
		is_cluster: boolean, tells if galaxy is in the cluster set
		is_other: boolean, tells if galaxy is in the OTHER set
		is_real: boolean, tells if galaxy is a real (observational) galaxy
		is_rsim: boolean, tells if galaxy is a Roediger simulation
		is_vsim: boolean, tells if galaxy is a Vollmer simulation
		is_ref: boolean, tells if galaxy is a reference observational galaxy
	
	For real galaxies, the following attributes are also set:
		xcenter: RA (right ascension) of center in degrees
		ycenter: DEC (declination) of center in degrees
		R25: a radius measure corresponding to the galaxy's stellar disk
		beam_PA: PA of the beam used to take the image, in arcsec
		beam_d1: first diameter of the beam used to take the image, in arcsec
		beam_d2: second diameter of the beam used to take the image, in arcsec
		A_beam: area in the beam, in arcsec^2
		channel_width: width of channels that made the moment-0 map
		rms_ch_jy_b: rms error per channel
		rms_for_noise_runs: controls amount of noise added to pixels for Monte Carlo simulation
	"""
	#try:
	#	for attr in ('basic_info','real_data','noise_data','zdata'):#'data_raw'
	#		setattr(Ginstance,attr,Ginstance._load_array(attr))
	#except IOError: pass
	filename,openname,filetype=Ginstance.filename,Ginstance.openname,Ginstance.filetype
	fileheader=get_raw_data_from_fits(Ginstance)
	wcs = WCS(openname)
	is_real, is_atlas3d, is_cluster, is_other, is_rsim, is_vsim = Ginstance.getattrs(
		'is_real','is_atlas3d', 'is_cluster', 'is_other', 'is_rsim', 'is_vsim'
	)
	
	if is_real:
		pix_scale_arcsec=abs(fileheader['CDELT1'])*3600
	else:
		pix_scale_arcsec=fileheader['CDELTS']
	
	coorsfloat=[]
	R25 = nan
	
	#print(f"\tfits_data: '{filename}': is_rsim={is_rsim}, is_vsim={is_vsim}, is_real={is_real}")
	
	if filename in cluster:
		key=(filename if (filename=='ic3392') else ('ngc'+filename))
		coorsdataline=coorslinesdata[key]
		for item in coorsdataline:
			try: coorsfloat.append(float(item))
			except ValueError: coorsfloat.append(item)
		xcenter=(abs(coorsfloat[1])+coorsfloat[2]/60+coorsfloat[3]/3600)*15
		if coorsfloat[1]<0: xcenter *= -1
		ycenter=(abs(coorsfloat[4])+coorsfloat[5]/60+coorsfloat[6]/3600)
		if coorsfloat[4]<0: ycenter *= -1
		
		line=np.array([nan]+shapedata[key][1:],dtype=float)
		
		PA,inclination=line[1:3]
		#try: shortside_med_extent=float(line[3])
		#except ValueError: pass
		#flux_per_chung=float(line[4])
		#flux_error_per_chung=float(line[5])
		#W20=float(line[6])
		
		beam_d1,beam_d2=line[8:10]
		d_beam_arcsec=d_beam_func(beam_d1,beam_d2)
		A_beam=beam_d1*beam_d2
		beam_per_pixel=A_beam/np.power(pix_scale_arcsec,2)
		
		#print 'A_beam = %g, pix_scale_arcsec=%g, beam_per_pixel=%g'%(A_beam, pix_scale_arcsec, beam_per_pixel)
		
		channel_width=(21 if filename in {'4606','4607'} else 10.4)
		
		rms_ch_jy_b=float(line[7]) #"rms per channel of the final cube imaged with robust=1" *(aj_138_6_1741.pdf)
		rms_for_noise_runs=(rms_ch_jy_b*beam_per_pixel)*channel_width/1000
		'''	to get rms divide by beam in units of pixels^2 and multiply by channel width,
				channel width=10.4 km/s except for 4606,4607
			then divide by 1000 (converts mJy-->Jy)'''
		"""if debug: print filename+'\t'+'\t'.join(np.array([beam_d1,beam_d2,A_beam,beam_per_pixel]).round(1).astype(str))"""
		R25=line[10]/2 #in arcmin
		#if not debug: print filename, u'PA=%g\N{DEGREE SIGN}' %PA, u'i=%g\N{DEGREE SIGN}' %inclination
		beam_PA=line[11]
		try: xpix, ypix, zpix = wcs.wcs_world2pix(xcenter,ycenter,0,0)
		except TypeError: xpix, ypix = wcs.wcs_world2pix(xcenter,ycenter,0)
# 		if filename=='1427a': xpix,ypix=58,64
	
	elif filename in OTHER:
		line=OTHER_data[filename.lower()]
		RAcoors,DECcoors=line[1],line[2]
		for r in ('h','m','s','d'):
			RAcoors,DECcoors=RAcoors.replace(r,' '),DECcoors.replace(r,' ')
# 		RAcoors,DECcoors=floatsplit(RAcoors),floatsplit(DECcoors)
		RAcoors,DECcoors = (list(map(float, c.split())) for c in (RAcoors,DECcoors))
		xcenter=(RAcoors[0]+RAcoors[1]/60+RAcoors[2]/3600)*15
		ycenter=(DECcoors[0]+DECcoors[1]/60+DECcoors[2]/3600)
		xpix, ypix, zpix = wcs.wcs_world2pix(xcenter,ycenter,0,0)
		PA,inclination=float(line[3]),float(line[4])
		beam_d1,beam_d2=(float(v) for v in line[6].split('&'))
		d_beam_arcsec,rms_ch_jy_b=d_beam_func(beam_d1,beam_d2),float(line[5])
		channel_width=2.58
		A_beam=beam_d1*beam_d2
		beam_per_pixel=A_beam/np.power(pix_scale_arcsec,2)
		rms_for_noise_runs=(rms_ch_jy_b*beam_per_pixel)*channel_width/1000
		beam_PA=0
		
	elif filename in ATLAS3D:
		try: line=ATLAS3D_data['ngc'+filename]
		except KeyError: line=ATLAS3D_data[filename.lower()]
		xcenter,ycenter=float(line[1]),float(line[2])
		xpix, ypix, zpix = wcs.wcs_world2pix(xcenter,ycenter,0,0)
		PA,inclination=float(line[3]),float(line[4])
		beam_d1,beam_d2=(float(v) for v in line[6].split('&'))
		d_beam_arcsec,rms_ch_jy_b=d_beam_func(beam_d1,beam_d2),float(line[5])
		channel_width=4
		A_beam=beam_d1*beam_d2
		beam_per_pixel=A_beam/np.power(pix_scale_arcsec,2)
		"""if debug: print filename+'\t'+'\t'.join(np.array([beam_d1,beam_d2,A_beam,beam_per_pixel]).round(1).astype(str))"""
		rms_for_noise_runs=(rms_ch_jy_b*beam_per_pixel)*channel_width/1000
		"""
		'beam_per_pixel' is now determined for ATLAS3D galaxies
		before it was marked as being undetermined
		why I am not sure
		"""
		#if not debug: print filename, u'PA=%g\N{DEGREE SIGN}' %PA, u'i=%g\N{DEGREE SIGN}' %inclination
		beam_PA=0
	
	elif is_rsim:
		PA=0
		if filename[0]=='x': inclination=0
		elif filename[0]=='y': inclination=90
		sim_header=fits.open(openname)[0].header
		xpix,ypix=abs(sim_header['CRVAL1']/sim_header['CDELT1']),abs(sim_header['CRVAL2']/sim_header['CDELT2'])
		pix_scale_arcsec,d_beam_arcsec=.1,1.2
		"""A_beam=np.pi*np.power(d_beam_arcsec,2)"""
	elif is_vsim:
		fcomp=filename.split('_')
		voll_galaxy=fcomp[0][1:]
		inclination=int(fcomp[1])
		disk_wind_angle=float(fcomp[2])
		frame=int(fcomp[3])
		sim_header=fits.open(openname)[0].header
		xpix, ypix = vollmer_center_coors[voll_galaxy]
		pix_scale_arcsec, d_beam_arcsec = vollmer_psa[voll_galaxy], 1.2
		PA=vollmer_PA[voll_galaxy]
		"""
		PA=36.
		inclination=v4522_inclinations[int(filename.split('_')[-2])]
		xpix,ypix=256,256
		pix_scale_arcsec,d_beam_arcsec=.1,1.2
		"""
	
	else:
		print(f"'{filename}': this galaxy is unknown")
	#print(f'\t{Ginstance.filename} `fits_data`: directory is <{Ginstance.directory}>')
	
	if is_real and (filename in FORCE_CENTER_PIXEL_COORDINATES):
		xpix,ypix=FORCE_CENTER_PIXEL_COORDINATES[filename]
	
	if is_real:
		iterator = MultiIterator(baseline_attributes, observational_only_attributes)
		
		if inclination > high_i_threshold:
			Ginstance.deny_m2()
	else:
		iterator = baseline_attributes
		Ginstance.deny_observational()
		
		if inclination > HIGH_I_THRESHOLD: Ginstance.deny_m2()
	
	for attr in iterator:
		try: # set the attr based on the value in this scope
			setattr(Ginstance,attr,eval(attr))
		
		except (UnboundLocalError, NameError):
			pass
		#	# if not defined in this scope, ensure that it is already defined on the instance
		#	try:
		#		getattr(Ginstance,attr)
		#	except AttributeError:
		#		setattr(Ginstance,attr,Ginstance.NULL)
	"""basic_info=(PA, inclination, xpix, ypix, pix_scale_arcsec, d_beam_arcsec)
	if is_real:
		real_data=(xcenter,ycenter,R25,beam_PA,beam_d1,beam_d2)
		try: noise_data=(A_beam, channel_width, rms_ch_jy_b, rms_for_noise_runs)
		except: noise_data=tuple(nan for v in noise_data_values)
	else: real_data,noise_data=(tuple(nan for v in s) for s in (real_data_values,noise_data_values))
	
	Ginstance.central_angle=nan
	Ginstance._setattrs_unzip(basic_info_values,basic_info)
	Ginstance._setattrs_unzip(real_data_values, real_data)
	Ginstance._setattrs_unzip(noise_data_values,noise_data)
	if Ginstance.save:
		for artype,ar in zip(('basic_info','real_data','noise_data'),(basic_info,real_data,noise_data)):
			Ginstance._save_array(artype,ar)"""

def _gal_check_association(filename):
	"""Return the sample corresponding to the provided filename (str).
	Note: here, filename should not name a single galaxy, but a sample.
	It may also be expressed using Python list syntax, i.e.
	"[gal1, gal2, ...]".
	"""
	if re.match('[[].*[]]',filename) is not None:
		return eval(filename[1:-1])
	fn=filename.lower()
	if fn in {'v','virgo'}: return Virgo
	if fn in {'c','cluster'}: return cluster
	if fn in {'at','atlas'}: return ATLAS3D
	if fn in {'h','high'}: return list(edgeon_ab.keys())
	if fn in {'l','low'}: return {gal for gal in cluster if gal not in edgeon_ab}
	if fn in {'o','other'}: return OTHER
	if 'ref' in fn: return external_references

	if 'sim' in fn:
		if 'x' in fn: return sim_fn_x
		if 'y' in fn: return sim_fn_y
		return sim_fn

	if 'v'==fn[0]:
		return get_vollmer_fn(fn)

	# elif fn=='v4522': return v4522
	# elif 'v4522' in fn: return v4522_collection[int(filename[-1])]
	# else: return None

def _gal_check_individual(filename):
	"""Given a galaxy name, make sure that it is properly formatted
	to be recognized by the program. Return a corrected version if needed."""
	fn=filename.lower()
	if 'v'==fn[0]:
		# match=re.search('v\d_\d+',fn)
		# if match: return match.group().replace('v','voll4522_')
		# match=re.search('voll4522_\d_\d+',fn)
		# if match: return match.group()
		return get_vollmer_fn(fn)
	if fn in galaxy_samples:
		return fn
	else:
		FN=filename.upper()
		if FN in galaxy_samples: return FN
		elif filename in galaxy_samples: return filename
		else: return None

def _gal_identify(filename):
	"""If `filename` denotes a sample of galaxies, return
	an iterable of galaxies in that sample; otherwise,
	return a set whose sole element is the name corresponding
	to that galaxy."""
	res = _gal_check_association(filename)
	if res is not None: return res
	
	res = _gal_check_individual(filename)
	if res is not None: return {res,}

def gal_check(override=None):
	"""Allows for specifying galaxy names from the command line."""
	if len(sys.argv)>1: override=sys.argv[1]
	if override is None:
		init_input = input('Which galaxy would you like to work with? ')
		if 'q' in init_input: sys.exit()
	else: init_input=override
	
	fn_split=init_input.split()
	asy_gal_list = reduce(list.__add__,[_gal_identify(name) for name in fn_split])
	
	#unrecognized = [i for g,i in zip (asy_gal_list, fn_split) if g is None]
	#if unrecognized:
	#	for unrec in unrecognized: print "input '%s' unrecognized"%unrec
	
	if asy_gal_list is None: sys.exit()
	
	names,original_indices=np.unique([g for g in asy_gal_list if g is not None],return_index=True)
	
	return names[np.argsort(original_indices)]

def errtrace(check=None, message=None):
	"""Prints out an error traceback message with some formatting.
	Meant to be used immediately after catching an exception,
	which supplies the traceback information via `traceback.format_exc`.
	"""
	err=traceback.format_exc().split('\n')[:-1]
	if check is None: pass
	else:
		check_count=0
		for p in check:
			if err[p[0]]==check[1]: check_count+=1
		if check_count==len(check):
			print(message)
			return None
	for i,t in enumerate(err):
		print(i,'----    ='+t)

def return_ref_data(filename,xcenter,ycenter):
	"""Return data corresponding to the radio deficit map
	for the sepcified galaxy having (xcenter,ycenter) center."""
	prefix='fits files/'
	try:
		if filename=='4569': fpath='n'+filename+'_sings_0.5_mask.fits'
		elif filename=='4402': fpath='N'+filename+'-RIRRAT-RACL07.FITS'
		else: fpath='n'+filename+'_0.5_mask.fits'
		
		fits_file_r=fits.open(prefix+fpath)
		wcs_r = WCS(prefix+fpath)
		rdata = fits_file_r[0].data
		try: pix_scale_r=abs(fits_file_r[0].header['CDELT1'])
		except KeyError: pix_scale_r=abs(fits_file_r[0].header['PLTSCALE']/3600)
		fits_file_r.close()
		#if filename=='4569': pix_scale_r=pix_scale*5/9
		pix_scale_arcsec_r=pix_scale_r*3600
		xpix_r, ypix_r = wcs_r.wcs_world2pix(xcenter,ycenter,0)
		return (rdata,pix_scale_arcsec_r,xpix_r,ypix_r)
	except IOError: return (None,)*4 #print filename,'return_ref_data error: ref_data not returned'

def sanitize_path(p):
	"""Replace non-permitted characters in filenames with dashes."""
	return re.subn('[/|:]','-', p)[0]

lprint=partial(print,sep='\n')

def copy_paper_symlink_tree(paper, link_folder):
	"""Given a paper name and associated folder link_folder, which
	expressed a path relative to the associated directory where figures
	for the paper are saved, copy the files referenced by the symlinks
	in the folder to another directory, maintaining the structure of the
	directory under `link_folder`.
	"""
	paper = str(paper)
	if not isinstance(link_folder, str):
		link_folder = os.path.join(*link_folder)
	
	inpath = os.path.join(get_figure_link_path(paper), link_folder)
	path = os.path.join(get_paper_path(paper), 'copytree', link_folder)
	if os.path.exists(path):
		os.remove(path)
	
	shutil.copytree(inpath, path)

__all__= ('load_array','save_array','save_fmt','merge_dicts','d_beam_func',#'select',
'print_update','makepath','makepath_from_file','touch_directory','savetxt',
'str_replace','fpull','coorslinesdata','shapedata','OTHER_data','ATLAS3D_data',
'listify','listify2','get_shortside_data','sanitize_path','lprint','cluster_shape_data',
'get_raw_data_from_fits','fits_data','gal_check','errtrace','return_ref_data',
'cluster_coor_data','copy_paper_symlink_tree')

if __name__=='__main__':
	res=gal_check()
	print(res)