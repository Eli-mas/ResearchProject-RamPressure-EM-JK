from .galaxy_attribute_information import all_attributes
__all__ = (
	'Galaxy',
	'all_attributes',
)


"""
attribute information:
PA = position angle (of major axis) in degrees
inclination = viewing (inclination) angle in degrees
xcenter, ycenter = RA, DEC
xpix, ypix = (x,y) center coordinates in galaxy momemt-0 FITS map
pix_scale_arcsec = size (scale) of each pixel in arcsec
"""