"""
02-18-19 removals: data pull,<smoothing block from file_return>
02-19-19: removed traces of THINGS
"""


import sys, os, os.path, traceback, re
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

# from comp.contiguous import reducer # circular import

from prop.galaxy_file import *
from prop.simprop import vollmer_center_coors, vollmer_psa, vollmer_PA
# from prop.simprop import v4522_inclinations
from prop.asy_prop import *
from prop.sim_fn_gen import sim_fn_x,sim_fn_y

from cls.adc_prep import basic_info_values, real_data_values, noise_data_values



"""def multi_membership(value,containers):
	return ((v in c) for c in containers)
def zip_membersip(test_values,containers):
	return ((v in c) for v,c in zip(test_values,containers))"""

#def np_print() # for printing arrays conveniently

def load_array(path, **kw): #ret_mmap=False,
# 	if ret_mmap: return np.load(path, mmap_mode='r', **kw)
	return np.load(path, **kw)

def save_array(path, arr):#, tests=False, full_tests=True, ret_mmap=False
	"""save numpy file (*.npy) of numpy array and assert correct loading via memory map"""
	arr = np.atleast_1d(arr)
	np.save(makepath_from_file(path), arr)
# 	if ret_mmap: mmap=np.load(path, mmap_mode='r')
# 	else: 
	return arr
# 	if tests:
# 		assert np.array_equal(arr, mmap)
# 		if full_tests:
# 			assert np.array_equal(arr.shape, mmap.shape)
# 			assert np.array_equal(arr+1.5, mmap+1.5)
# 			assert np.array_equal(arr-1.5, mmap-1.5)
# 			assert np.array_equal(arr*1.5, mmap*1.5)
# 			assert np.array_equal(arr/1.5, mmap/1.5)
# 			index=np.arange(len(arr)+1)%len(arr)
# 			assert np.array_equal(arr[index], mmap[index])
# 			assert np.array_equal(np.where(arr), np.where(mmap))
# 	return mmap

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
		#dstr=data.to_string().split('\n')
		f.write('\n'.join(dstr[1:]))

# def select(test_value,input,output=None):
# 	if output is None: comp=input
# 	else: comp=zip(input,output)
# 	for i,o in comp:
# 		if i==test_value: return o

def merge_dicts(*dicts,**k):
	d0={}
	consume(d0.update(d) for d in dicts)
	if k: d0.update(k)
	return d0

def d_beam_func(d1,d2): return np.sqrt(d1*d2)

# def print_stack(stack,index=None,item=None,exclude=[],joiner='   '):
# 	"""
# 	'stack' should be the result of calling inspect.stack()
# 	"""
# 	exclude=set(exclude)
# 	if index is None:
# 		if item is None: 
# 			for i,l in enumerate(stack):
# 					print('\t stack item %i'%i)
# 					for j,e in enumerate(l):
# 						print('\t\t%i %s'%(j,e))
# 		else:
# 			item=listify2(item)
# 			sp=(l[j] for j in item for l in stack)
# 			sp=(e for e in sp if e not in exclude)
# 			print(joiner.join(sp))
# 		#print	
# 	else:
# 		for i in listify2(index):
# 			stack_element=stack[i]
# 			if item is None: 
# 				print('\t stack item %i'%i)
# 				for j,e in enumerate(stack_element):
# 					print('\t\t%i %s'%(j,e))
# 			else:
# 				sp=(stack_element[j] for j in listify2(item))
# 				sp=(e for e in sp if e not in exclude)
# 				print(joiner.join(sp))
# 		#print
# 	return ''

def print_update(*p):
	#stdout.write("\r"+str(p))
	#stdout.write('\x1b[2K\r'+' '.join([str(i) for i in p]))
	#stdout.flush()
	print('\x1b[2K\r'+' '.join([str(i) for i in p]),flush=True,end='')

"""
def print_clear(): print_update('')

def print_io(stream,*p,**k):#,pr=True
	try: pr=k['pr']
	except KeyError: pr=True
	t=' '.join(str(e) for e in p)
	if pr: print(t)
	stream.write(t+'\n')
"""

def makepath(p): #https://stackoverflow.com/questions/273192/
	"""
	make a directory path, creating intermediary directories as required
	:param p:
	:return:
	"""
	path=os.path.join(os.getcwd(),p)
	try: os.makedirs(path)
	except OSError:
		if not os.path.isdir(path): raise
	return path

def makepath_from_file(f):
	"""
	make a directory to host a file given the full file path
	:param f: full file path
	:return:
	"""
	makepath(f[:len(f)-f[::-1].index('/')])
	return f

def touch_directory(directory):
	"""
	'touch' a directory -- make the directory's modification date (mtime)
	change by adding and removing a file. There are other ways to do this, but
	this is a simple way.
	:param directory: directory where the file is created
	:return:
	
	
	! ! ! Much simpler approach--
		use the Unix-native `touch` command with `-m` argument !
	"""
	f = '__temp__{}'
	i=0
	while True:
		t = os.path.join(directory,f.format(i))
		#print(t)
		if not os.path.exists(t): break
		i+=1
	try:
		with open(t,'x') as f: pass
		os.remove(t)
	finally:
		if os.path.exists(t):
			os.remove(t)

def savetxt(p,ar,**k):
	makepath_from_file(p)
	np.savetxt(p,ar,**k)

"""def makefile(filepath,*args,**kw):
	k=dict(saving_function=np.savetxt)
	k.update(kw)
	saving_function=k['saving_function']
	k.pop('saving_function')
	fs=filepath.split('/')
	makepath('/'.join(fs[:-1]))
	saving_function(*args,**k)"""



def str_replace(s,args,r=''):
	for arg in args: s=s.replace(arg,r)
	return s

def fpull(file,c=None,a=[],s=0,e=0,rep=None,r=None,ret=0):
	f=open(file,errors='replace')
	if not e: l=f.readlines()[s:]
	else: l=f.readlines()[s:e]
	if rep==None: l=[pl.split() for pl in l]
	else: l=[str_replace(pl,rep,r=r).split() for pl in l]
	if c==None: pass
	else: l=[pl for pl in l if len(pl)>c]
	if not a: pass
	else:
		for pl in l: pl[a[1]]=str_replace(pl[a[1]],a[0])
	f.close()
	if not ret:
		d={}
		for pl in l: d[pl[0].lower()]=pl
		return d
	elif ret=='a': return np.array(l)

#edge_raw=fpull(MAIN_PROGRAM_DIRECTORY+'graph_edge.list')
#central_raw=fpull(MAIN_PROGRAM_DIRECTORY+'gal_center_angles.list')
coorslinesdata=fpull(DATA_SOURCES_PATH+'gal_coor.file',rep=('h','m','s','d'),r=' ')
shapedata=fpull(DATA_SOURCES_PATH+'gal_shape_mod.file')
OTHER_data=fpull(DATA_SOURCES_PATH+'OTHER/OTHER.file',s=1,e=2)

ATLAS3D_data=fpull(DATA_SOURCES_PATH+'ATLAS3D/ATLAS3D.file',s=1,e=24)
#ftype_dic={'s': 'sim', 'r': 'real','vs':'v4522'}

# p_suf=('cent','nc','m2c','m2nc')
# sc_suf=('extsc','flsc','htsc','qextsc','qflsc','m2extsc','m2flsc')
# b_suf=('int','out','m2int','plad')
# p_suf_cart=('cent_cart','nc_cart','m2c_cart','m2nc_cart')
# suffices=[p_suf,sc_suf,b_suf,p_suf_cart]

def listify(input,convert): return [input] if isinstance(input,convert) else input
def listify2(input,convert=list):
	#return [input] if (not isinstance(input,convert)) else input
	try: return list(input)
	except TypeError: return [input]

def get_shortside_data(Ginstance):
	shortside_list_graph=np.array(Ginstance.shortside_list_graph)
	
	if deny_m2_high_i and Ginstance.inclination>high_i_threshold:
		shortside_show=shortside_list_graph
		shortside_median=nan
	else:
		shortside_show=np.array(Ginstance.shortside_list_graph_deproject)
		shortside_median=np.median(shortside_show[:,1])
		shortside_show[:,1]/=shortside_median
	
	shortside_show[:,0]=shortside_show[:,0]%360
	
	return shortside_show,shortside_median

'''
def file_return(filename):
	"""returns the information required to access data for a given galaxy"""
	if 'x' in filename or 'y' in filename:
		filetype='s'
		if filename not in sim_fn:
			sim_name_file=open(MAIN_PROGRAM_DIRECTORY+'sim_names.file','a')
			sim_name_file.write(' %s'%filename)
			sim_name_file.close()
			sim_fn.append(filename)
		#sim_name=DATA_SOURCES_PATH+'sim_fits/'+'sigma_gas_'+filename[0]+'_0'+filename[1:]+'.fits'#os.path.join(os.getcwd(),)
		#openname=DATA_SOURCES_PATH+'sim_fits/'+'sigma_gas_'+filename[0]+'_0'+filename[1:]+'_smooth-test.fits'#os.path.join(os.getcwd(),)
		sim_name = RSIM_FITS_SOURCE_FILE.format(rtype=filename[0], index=filename[1:])
		openname = RSIM_FITS_PROCESSED_FILE.format(filename)
	elif 'v'==filename[0]:
		filetype='vs'
		fcomp=filename.split('_')
		openname=VSIM_FITS+'{group}/smoothed__{gal}.fits'.format(group='_'.join(fcomp[:-1]),gal=filename)
	else:
		filetype='r'
		if filename in OTHER:
			path1=DATA_SOURCES_PATH+'OTHER/'+filename+'.fits'
			path2=DATA_SOURCES_PATH+'OTHER/'+filename+'.FITS'
		elif filename in ATLAS3D:
			path1=DATA_SOURCES_PATH+'ATLAS3D/Serra2012_Atlas3D_Paper13/all_mom0/NGC'+filename+'_mom0.fits'
			path2=DATA_SOURCES_PATH+'ATLAS3D/Serra2012_Atlas3D_Paper13/all_mom0/UGC'+filename+'_mom0.fits'
		else:
			path1=DATA_SOURCES_PATH+'fits files/ngc'+filename+'.mom0.fits'
			path2=DATA_SOURCES_PATH+'fits files/'+filename+'.mom0.fits'
		openname=path1 if os.path.exists(path1) else path2
	return [filename,openname,filetype]
'''	
# get_zero, zdata_ret moved to comp.contiguous

def get_raw_data_from_fits(Ginstance):
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
	"""returns relevant data pertaining to galaxy and corresponding fits file"""
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
		xcenter=(coorsfloat[1]+coorsfloat[2]/60+coorsfloat[3]/3600)*15
		ycenter=(coorsfloat[4]+coorsfloat[5]/60+coorsfloat[6]/3600)
		
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
		if filename=='1427a': xpix,ypix=58,64
	
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
		
		if inclination >= high_i_threshold:
			Ginstance.deny_m2()
	else:
		iterator = baseline_attributes
		Ginstance.deny_observational()
		
		if inclination == 90: Ginstance.deny_m2()
	
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
	if re.match('[[].*[]]',filename) is not None: return eval(filename[1:-1])
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
	res = _gal_check_association(filename)
	if res is not None: return res
	
	res = _gal_check_individual(filename)
	if res is not None: return {res,}

def gal_check(override=None):
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

def errtrace(check=None,message=None):
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
	return re.subn('[/|:]','-', p)[0]

lprint=partial(print,sep='\n')

__all__= ('load_array','save_array','save_fmt','merge_dicts','d_beam_func',#'select',
'print_update','makepath','makepath_from_file','touch_directory','savetxt',
'str_replace','fpull','coorslinesdata','shapedata','OTHER_data','ATLAS3D_data',
'listify','listify2','get_shortside_data','sanitize_path','lprint',
'get_raw_data_from_fits','fits_data','gal_check','errtrace','return_ref_data')

if __name__=='__main__':
	res=gal_check()
	print(res)