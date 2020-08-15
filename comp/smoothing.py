from functools import partial

import numpy as np

from astropy.io import fits
#import astropy.convolution as ac
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma

import prop.simprop as simprop

from asy_io.asy_paths import RSIM_FITS_SOURCE_FILE, RSIM_FITS_PROCESSED_FILE, VSIM_SOURCE, VSIM_FITS_SMOOTHED_FILE, VSIM_FITS_RAW_FILE
from asy_io.asy_io import makepath_from_file, print_update
from prop.asy_prop import sim_back_cutoff
from prop.simprop import vollmer_psa

# INITIAL = 

# from https://mail.scipy.org/pipermail/astropy/2014-October/003464.html
def rsmoother(file, res=1.2, pr=False):
	"""
	smoother function intended for roediger files

	see:
	https://astropy-tutorials.readthedocs.io/en/latest/rst-tutorials/synthetic-images.html
	`gaussian_sigma_to_fwhm` in Constants section of astropy.stats docs, 
		https://astropy.readthedocs.io/en/latest/stats/index.html
	"""

	print_update(file)
	filename=RSIM_FITS_SOURCE_FILE.format(rtype=file[0], index=file[1:])
	cutoff = sim_back_cutoff[file[0]]
	processed_file = RSIM_FITS_PROCESSED_FILE.format(file)

	fits_file=fits.open(filename)
	header_o=fits_file[0].header
	fits_file.close()

	fwhm=res/abs(header_o['CDELT1'])

	data=fits.getdata(filename)
	smoothed=convolve(data, Gaussian2DKernel(stddev =fwhm * gaussian_fwhm_to_sigma))
	smoothed[smoothed<cutoff] = 0
	hdu=fits.PrimaryHDU(smoothed)
	hdulist = fits.HDUList([hdu])
	hdulist.writeto(processed_file, overwrite=True)

	file_s=fits.open(processed_file)
	header_s=file_s[0].header
	header_s['CDELTS']=res
	header_s.append('ORIGDATA')
	for h in header_o:
		try: header_s[h]=header_o[h]
		except ValueError:
			if pr: print('Error:', header_o[h])
			else: pass

	file_s.writeto(processed_file, overwrite=True)
	file_s.close()

	if pr: print(f'{file}: smoothed file created')

def vtranspose(galaxy, ind):
	return np.moveaxis(
		fits.getdata(VSIM_SOURCE.format(gal=galaxy, ind=ind)), 
		2, 0
	)

def write_fits(savepath, data, _fields={}, **kw):
	hdu=fits.PrimaryHDU(data)
	hdulist = fits.HDUList([hdu])
	hdulist.writeto(savepath, overwrite=True)

	if (_fields or kw):
		file_s=fits.open(savepath)
		header_s=file_s[0].header

		for dictionary in (_fields, kw):
			for k, v in dictionary.items(): header_s[k]=v
		file_s.writeto(savepath, overwrite=True)

		file_s.close()


def smooth_fits(data, original_res, output_res, cutoff=None):
	fwhm=output_res/original_res
	smoothed=convolve(data, Gaussian2DKernel(stddev =fwhm * gaussian_fwhm_to_sigma))
	if cutoff is not None: smoothed[smoothed<cutoff]=0
	return smoothed

def vsmoother(galaxy, index, res=1.2, vsim_cutoff=.1, skip_first=True, raw=False):
	res_o = vollmer_psa[galaxy]#0.1,
	if raw: outstr = VSIM_FITS_RAW_FILE
	else: outstr = VSIM_FITS_SMOOTHED_FILE

	outpath=partial(outstr.format, 
		gal=galaxy, 
		inc=sorted(simprop.vollmer_data[galaxy].keys(), reverse=True)[index], 
		disk_wind=simprop.vollmer_disk_wind_inclinations[galaxy]
	)

	data=vtranspose(galaxy, index)
	if skip_first: data=data[1:]

	if raw:
		for i, d in enumerate(data):
			print_update('vsmoother (raw): {gal}-{index}  frame {frame}'.format(gal=galaxy, index=index, frame=i))
			savepath = outpath(frame=i)
			makepath_from_file(savepath)
			write_fits(savepath, d, TSTEP='--', CDELTO=res_o, CDELTS=res, CUTOFF=vsim_cutoff)
	else:
		for i, d in enumerate(data):
			print_update('vsmoother: {gal}-{index}  frame {frame}'.format(gal=galaxy, index=index, frame=i))
			smoothed=smooth_fits(d, original_res=res_o, output_res=res, cutoff=vsim_cutoff)
			savepath=outpath(frame=i)
			makepath_from_file(savepath)
			write_fits(savepath, smoothed, TSTEP='--', CDELTO=res_o, CDELTS=res)
	print_update('')

# __all__ = ()
if __name__=='__main__':
	print('smoothing of roediger and vollmer files has been done already; exiting')

	for galaxy in vollmer_psa:
		for i in range(3):
			print('vsmoother(%s, %i)'%(galaxy, i))
			vsmoother(galaxy, i)

	print('smoothing finished')