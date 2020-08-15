__all__=(
    'galaxy_plot_funcs', 'galaxy_fig_funcs', 'galaxyseries_timeplot_funcs',
    'galaxyseries_plot_funcs', 'galaxyseries_fig_funcs',
    'galaxyseries_smoothing_angle_plot_funcs', 'galaxyseries_anglar_map_funcs'
)

for m in __all__:
	exec(f'from . import {m}')

del m