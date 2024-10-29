import matplotlib.pyplot as plt
plt.style.use('/home/gridsan/gschuette/universal/matplotlib/plot_style.txt')
import matplotlib.patheffects as pe
import sys
sys.path.insert(0,'../data_utils/')
from HiCDataset import HiCDataset

class MapCompare:

    def __init__(
        self,
        cooler_IMR90='../../data/outside/IMR90_hg19.mcool',
        cooler_GM12828='../../data/outside/GM12878_hg19.mcool',
        resolution=20_000,
        region_length=1_280_000
    ):

        self.imr = HiCDataset(cooler_IMR90,resolution)
        self.gm = HiCDataset(cooler_GM12828,resolution)
        self.resolution = resolution
        self.region_length = region_length

    def __call__(
        self,
        chrom=None,
        start_idx=None,
        region_length=None,
        balance=True,
        adaptive_coarsegraining=False,
        adaptive_coarsegraining_cutoff=3,
        adaptive_coarsegraining_max_levels=8,
        interp_nans=False,
        vmin=None,
        vmax=1,
        cmap='fall'
    ):

        if start_idx is None:
            stop = None
        elif region_length is None:
            stop = start_idx + self.region_length
        else:
            stop = start_idx + region_length
        
        gm_map = self.gm.fetch(
            chrom,
            start_idx,
            stop,
            balance,
            adaptive_coarsegraining,
            adaptive_coarsegraining_cutoff,
            adaptive_coarsegraining_max_levels,
            interp_nans
        )

        imr_map = self.imr.fetch(
            chrom,
            start_idx,
            stop,
            balance,
            adaptive_coarsegraining,
            adaptive_coarsegraining_cutoff,
            adaptive_coarsegraining_max_levels,
            interp_nans
        )

        fig,ax,im,cbar = imr_map.plot_with(gm_map,vmin=vmin,vmax=vmax,cmap=cmap)

        num_bins = ax.get_xlim()[-1]
        
        ax.text(num_bins-1,.01*num_bins,'IMR90',horizontalalignment='right',verticalalignment='top',
                color='black',path_effects=[pe.withStroke(linewidth=1, foreground="white")])
    
        ax.text(.01*num_bins,.995*num_bins-1,'GM12878',horizontalalignment='left',verticalalignment='baseline',
                color='black',path_effects=[pe.withStroke(linewidth=1, foreground="white")])

        if chrom is not None:
            title = f'Chromosome {chrom}'
            if start_idx is not None:
                title+= ':\n'
                title+= f'{start_idx:,}-{stop:,}'

            ax.set_title(title)

        return fig, ax, im, cbar
        

###################################
# Make plot

print('Initialize object')
mc = MapCompare()
import matplotlib as mpl

scale = 130_000_000 / 4_500_000#3_000_000
mpl.rcParams['figure.figsize'] = (scale*7.2,scale*4.45)

print('Make figure')
fig, ax, im, cbar = mc(chrom='11')

lims = ax.get_xlim()
length = lims[-1]

alpha=.5
linewidth=.5 * (3_000_000 / 130_000_000)
color = 'k'
ax.plot([0,length-1_256_000//20_000],[1_256_000//20_000,length],color=color,alpha=alpha,linewidth=linewidth)
ax.plot([1_256_000//20_000,length],[0,length-1_256_000//20_000],color=color,alpha=alpha,linewidth=linewidth)

ax.plot([0,length-2*1_256_000//20_000],[2*1_256_000//20_000,length],color=color,alpha=alpha,linewidth=linewidth)
ax.plot([2*1_256_000//20_000,length],[0,length-2*1_256_000//20_000],color=color,alpha=alpha,linewidth=linewidth)

ax.set_xlim(lims)
ax.set_ylim(lims[-1::-1])

# Save figure
print('Save figure')
fig.savefig('GM_vs_IMR_chrom_11.pdf')
