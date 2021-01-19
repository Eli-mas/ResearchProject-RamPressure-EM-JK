"""This module defines names that correspond to scalar attributes
on the Galaxy class."""
from core import MultiIterator

"""
non_arrays = frozenset((
	'A_beam', 'EA', 'EA_trig', 'ER', 'ER_trig', 'FA', 'FA_rw', 'FA_trig', 'FA_trig_rw', 'FR', 'FR_rw',
	'FR_trig', 'FR_trig_rw', 'HTA', 'HTA_rw', 'HTA_trig', 'HTA_trig_rw', 'HTR', 'HTR_rw', 'HTR_trig',
	'HTR_trig_rw', 'PA', 'R25', 'TA', 'beam_PA', 'beam_d1', 'beam_d2', 'center_size', 'centerflux',
	'centroid_angle', 'centroid_radius', 'channel_width', 'd_beam_arcsec', 'dweighted_centroid_angle', 
	'dweighted_centroid_radius', 'edge_radius',
	'fweighted_centroid_angle', 'fweighted_centroid_radius', 'inclination', 'm2ER', 'm2FR',
	'm2_FluxAngle', 'm2ext_angle', 'noncenter_size', 'noncenterflux', 'pix_scale_arcsec',
	'qEA', 'qEA_trig', 'qER', 'qER_trig', 'qFA', 'qFA_rw', 'qFA_trig', 'qFA_trig_rw', 'qFR', 'qFR_rw',
	'qFR_trig', 'qFR_trig_rw', 'rms_ch_jy_b', 'rms_for_noise_runs', 'shortsum', 'shortsum_lin',
	'shortsum_trig', 'tail_A', 'tail_r', 'total_size', 'totalflux', 'weighted_centroid_angle',
	'weighted_centroid_radius', 'xcenter', 'xpix', 'ycenter', 'ypix',
))

observational_only_attributes=frozenset((
	'xcenter', 'ycenter', 'R25', 'beam_PA', 'beam_d1', 'beam_d2',
	'A_beam', 'channel_width', 'rms_ch_jy_b', 'rms_for_noise_runs',
))

baseline_attributes=frozenset((
	'PA', 'inclination', 'xpix', 'ypix', 'pix_scale_arcsec', 'd_beam_arcsec',
	'is_atlas3d', 'is_cluster', 'is_other', 'is_real', 	'is_rsim', 'is_vsim', 'is_ref',
	'directory', 'filename', 'filetype', 'openname',
))
"""


quantities = dict(
	basic_angles_ratios=(
		'FR', 'ER', 'EA', 'FA', 'qER', 'qFR',
		'HTR', 'HTA', 'TA', 'qEA', 'qFA', 'tail_A'),
	trig_q=(
		'EA_trig', 'FA_trig','ER_trig', 'FR_trig', 'HTA_trig', 'HTR_trig',
		'qEA_trig', 'qFA_trig', 'qER_trig', 'qFR_trig'),
	rw_quantities=(
		'HTR_trig_rw', 'qFR_trig_rw', 'qFA_rw', 'qFA_trig_rw', 'qFR_rw', 'FA_rw',
		'FA_trig_rw', 'FR_rw', 'FR_trig_rw', 'HTA_rw', 'HTA_trig_rw', 'HTR_rw'),
	
	m2_quantities=('m2ER', 'm2FR','m2_FluxAngle', 'm2ext_angle'),
	shortsums=('shortsum', 'shortsum_trig', 'shortsum_lin'),
	center_q = ('center_size', 'centerflux'),
	noncenter_q = ('noncenter_size', 'noncenterflux'),
	total_q = ('total_size', 'totalflux'),
	tail_centroid = (
		'weighted_centroid_angle', 'tail_r', 'weighted_centroid_radius',
		'centroid_angle', 'centroid_radius', 'dweighted_centroid_angle',
		'dweighted_centroid_radius', 'fweighted_centroid_angle', 'fweighted_centroid_radius'),
	other=('edge_radius',)
)


observational_only_attributes=frozenset((
	'xcenter', 'ycenter', 'R25', 'beam_PA', 'beam_d1', 'beam_d2',
	'A_beam', 'channel_width', 'rms_ch_jy_b', 'rms_for_noise_runs',
))

baseline_attributes=frozenset((
	'PA', 'inclination', 'xpix', 'ypix', 'pix_scale_arcsec', 'd_beam_arcsec',
	'is_atlas3d', 'is_cluster', 'is_other', 'is_real', 	'is_rsim', 'is_vsim', 'is_ref',
	'directory', 'filename', 'filetype', 'openname',
))





#non_arrays = frozenset(
#	MultiIterator(
#		observational_only_attributes,
#		baseline_attributes,
#		MultiIterator(*quantities.values())
#	)
#)
non_arrays = frozenset(MultiIterator(*quantities.values()))


__all__ = ('non_arrays','observational_only_attributes','baseline_attributes')

if __name__=='__main__':
	print(non_arrays)
	print(observational_only_attributes)
	print(baseline_attributes)