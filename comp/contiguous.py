"""The asymmetry methods for this project depend on having a map
of a galaxy's gas. We obtain this from moment-0 maps. This module
supplies pre-processing methods for reading these maps: in particular,
locating the body of gas that is contiguous with the pixel identified
as the galaxy's center, and removing from the map all pixels that are
not contiguous with this region."""

import time, sys, traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_erosion
"""from computation_functions import c_to_p"""
import traceback

#INITIAL=




binary_structure=generate_binary_structure(2, 2)

def reducer(fdata,xpix,ypix,simple_return=False):
	"""
	Given unreduced zdata, reduce it:
	i.e., isolate the contiguous region in the map centered
	around the galaxy's center coordinate,
	and zet to 0 all other contiguous groups in the image not connected
	to the central group.
	"""
	try:
		xc,yc=(int(np.round(v)) for v in (xpix,ypix))
	
		#ardata=np.array(fdata).astype('float')
		ardata = fdata # fdata should always be of float dtype
		
		# labels is an array with the same shape as ardata
		# where in place of the original data,
		# an integer is assigned indicating the contiguous group
		# to which the value at the given coordinate belongs
		# numLabels indicates the number of groups found
		labels, numLabels = label(ardata)
		
		# flux value of center coordinate
		fcent=fdata[yc][xc]
		# if this pixel has non-0 value, it is part of the target group
		if fcent!=0: center_label=labels[yc][xc]
		else:
			# if this is 0, then the central pixel is not
			# labeled as part of the target group,
			# and we have to find the target group;
			# do so by incrementing the coordinate in the x-direction,
			# and assume we will reach another pixel that is part of the target group
			add=0
			loc=False
			while not loc:
				if fdata[yc][xc+add]!=0:
					center_label=labels[yc][xc+add]
					fcent=fdata[yc][xc+add]
					loc=True
				else: add+=1
			#print 'new flux test: f(center)=%g'%fcent
		
		assert center_label != 0, \
			'contiguous.reducer: label of center region is 0'
		
		#cdata=np.copy(labels)
		#cdata[cdata!=center_label]=0
		#cdata[cdata==center_label]=1
		# below is a simpler way of doing this
		#cdata = (labels==center_label)#.astype(int)
		
		if simple_return:
			return (labels==center_label)*ardata
		
		cdata = (labels==center_label)#.astype(int)
		cdata_excl=np.copy(labels)
		cdata_excl[cdata_excl==center_label]=0
		cdata_excl[cdata_excl!=0]=1
		
		excl_labels=[l for l in range(numLabels) if l!=center_label]
		cdes_labeled_arrays=[]
		for l in excl_labels:
			cdes=np.copy(labels)
			cdes[cdes!=l]=0
			cdes_labeled_arrays.append(cdes)
		
		else: return cdata*ardata, cdata_excl*ardata, cdes_labeled_arrays, excl_labels, cdata, cdata_excl
	except IndexError:
		print(traceback.format_exc())

def get_zero(data_array):
	"""
	returns the background level of an array,
	taken as the most common value in the array,
	under the assumption that this value is not affected by noise:
	"""
	vals,counts=np.unique(data_array,return_counts=True)
	return vals[np.argmax(counts)]

def zdata_ret(Ginstance,data=None):
	"""
	generates zdata from data;
	zdata is the data array with the background value subtracted
	
	map_check relevant lines in former versions
	"""
	#print(f'\nzdata ret called on {Ginstance}\n')
	if data is None:
		data=Ginstance.data_raw
	
	try:
		"""zdata=data-get_zero(data)
		
		#if filename[0] in ('x','y'): #.astype(float).tolist()
		#else:
		zdata=reducer(zdata,Ginstance.xpix,Ginstance.ypix,simple_return=True)
			#if filename=='4396':
				#print 'pixel value:',zdata[113][109]
				#zdata[113][109]=0
		
		return zdata"""
		return reducer(data-get_zero(data), Ginstance.xpix, Ginstance.ypix, simple_return=True)
	
	except:
# 		errtrace()
		print(traceback.format_exc())
		print('%s zdata_ret error'%Ginstance.filename)
		return None

def galaxy_extend(zdata_array):
	"""
	NOTE THAT THIS WAS RENAMED: PREVIOUS NAME WAS 'galaxy_noise_extend'
	"""
	return (binary_dilation(zdata_array,structure=binary_structure).astype(bool) & (~zdata_array.astype(bool)))

def isolate_border_pixels(zdata):
	res=(zdata.astype(bool) & ~binary_erosion(zdata,structure=binary_structure).astype(bool))
	return (label(res,structure=binary_structure)[0]==1)
"""
def get_average_flux_at_coors(data,xc,yc,span,xpix,ypix):
	'''
	xc,yc specify the coordinates
	span gives the diameter of the box; it is expected to be odd
	'''
	halfspan=(span-1)//2
	X,Y=np.meshgrid(*([np.arange(-halfspan,halfspan+1,1)]*2))
	rows,cols=data.shape
	data_expand=np.full([rows+2*halfspan,cols+2*halfspan],np.nan);
	data_expand[halfspan:rows+halfspan,halfspan:cols+halfspan]=data
	#print data_expand
	
	return np.array([np.nanmean(data_expand[Y+y+halfspan,X+x+halfspan]) for x,y in zip(xc,yc)])[np.argsort(c_to_p(y,x,xpix,ypix)[0])]#

def get_average_flux_at_outer_coors(zdata,span=3):
	yc,xc=np.where(isolate_border_pixels(zdata))
	print len(xc),len(yc)
	return get_average_flux_at_coors(zdata,xc,yc,span=span)
"""
__all__=('reducer',)#'galaxy_extend','isolate_border_pixels'
