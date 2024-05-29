import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap as cm
import colorsys

# Typesetting
from typing import Any
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

@dataclass
class Modulation:
    q: list[Any]
    phi0: float
    Mf: list[list[Any]]

    def __post_init__(self):
        assert np.array(self.q).shape == (3,) # shape of the modulation vector must be 3
        assert np.array(self.Mf).shape == (3,2) # shape of the Fourier components must be 3

    @property
    def qx(self):
        return self.q[0]
    
    @property
    def qy(self):
        return self.q[1]
    
    @property
    def qz(self):
        return self.q[2]
    
    @property
    def Mcos_vec(self):
        M = np.array(self.Mf, dtype=float)
        return M[:,0]
    
    @property
    def Msin_vec(self):
        M = np.array(self.Mf, dtype=float)
        return M[:,1]
    
    # def __str__(self) -> str:
    #     return self.__repr__()

@dataclass
class MSSG:
    number_str: str
    name: str
    Mf_str: str
    gamma: float = np.nan
    index: tuple[int, ...] = ()
    modulations: list[Modulation] = None # type: ignore
    free_params: list[str] = None # type: ignore

    # I want to have a constructor only from number, name and Mf.
    # Then we have to generate the rest
    def __post_init__(self):
        # Unpack index to numbers
        self.index = tuple([int(x) for x in self.number_str.replace('m','').split('.')])

        if 74 < self.index[0] <= 142:   # tetragonal
            self.gamma = 90
        if 142 < self.index[0] <= 194:   # trigonal and hexagonal
            self.gamma = 120


        # Number and type of ks depends on index[2]
        k_types = {
            57:( ('a','b','0'), ('-b','a','0')),
            63:( ('a','0','0'), ('0','a','0')),
            68:( ('a','a','0'), ('-a','a','0')),
            80:( ('a','b','0'), ('-a-b','a','0'), ('b','-a-b','0')),
            82:( ('a','0','0'), ('-a','a','0'), ('0','-a','0')),
            83:( ('a','a','0'), ('-2*a','a','0'), ('a','-2*a','0')),
        }
        ks = k_types[self.index[2]]

        # Handling the string of fourier coefficients
        remove_chars = {x:'' for x in '(){}'}
        mfs = self.Mf_str.replace('},{',':').translate(str.maketrans(remove_chars))

        fmf = [[[a.strip() for a in x.split(',')] for x in mf.split(':')] for mf in mfs.split(';')]
        fmf = [[[a if a!='-' else '0' for a in x] for x in mf ] for mf in fmf]

        # Making modulations
        self.modulations = []
        for k,mf in zip(ks, fmf):
            self.modulations.append( Modulation(q=k, Mf=mf, phi0=0) )

        # determining free parameters
        # 1. In k from `ks` variable
        # 2. In magnetization fourier components
        self.free_params = []
        for x in 'ab':
            if x in ks[0]:
                self.free_params.append(x)

        for x in 'ABCDEF':
            if x in self.Mf_str:
                self.free_params.append(x)

    def __repr__(self) -> str:
        ret  = f'MSSG(number_str={self.number_str}, name={self.name}, index={self.index}\n'
        ret += f' free_params={self.free_params},\n '
        ret += ',\n '.join([str(mod) for mod in self.modulations])
        ret += '\n)'
        return ret
    
    def eval(self, tr_tab: dict[str, float]) -> list[Modulation]:
        set_fp = set(self.free_params)

        missing_symbols = set_fp - tr_tab.keys()

        if missing_symbols:
            raise KeyError(f'{missing_symbols} symbols are missing to evaluate modulations.')

        modulations_evaluated = []
        tt = str.maketrans({key:str(val) for key,val in tr_tab.items()})
        for mod in self.modulations:
            k_vals = [eval(x.translate(tt)) for x in mod.q]
            Mf_vals = [[eval(x.translate(tt)) for x in ma] for ma in mod.Mf]

            modulations_evaluated.append(Modulation(q=k_vals, Mf=Mf_vals, phi0=mod.phi0))

        return modulations_evaluated

def load_MSSG_tables() -> dict:
    with open('./MSSG.txt','r') as ff:
        lines = ff.readlines()

    mssg_tables = {}
    for line in lines:
        # Ignore comented lines or lines with not enough entries
        if (len(line) < 10) or (line[0] == '#'):
            continue

        # First split into name entries and fourier components
        it_mf = line.find('{') - 1

        mssg_number, mssg_name = line[:it_mf].split()[:2]
        mssg_mf = line[it_mf:].strip()

        mssg_tables[mssg_number] = MSSG(number_str=mssg_number, name=mssg_name, Mf_str=mssg_mf)

    return mssg_tables

MSSG_tables = load_MSSG_tables()

class MagneticLattice(object):
    '''Lattice with magnetic moments on each lattice point'''
    # Lattice defining properties
    gamma: float
    modulations: list[Modulation]
    # Lattice points
    lattice_ab: tuple[NDArray, NDArray]
    lattice_xy: tuple[NDArray, NDArray]
    magnetization_abc: tuple[NDArray, NDArray, NDArray]
    magnetization_xyz: tuple[NDArray, NDArray, NDArray]

    def __init__(self, N: int, modulations: list[Modulation], Nsub: int=1, gamma: float=90):
        self.gamma = gamma
        self.modulations = modulations

        # Na = Nb = lattice_extent
        na = np.arange(-N, N+1, Nsub)
        nb = np.arange(-N, N+1, Nsub)
        A, B = np.meshgrid(na,nb)
        self.lattice_ab = (A, B)
        self.lattice_xy = self.uvw2xyz(gamma, self.lattice_ab[0], self.lattice_ab[1])

        self.magnetization_abc = self.magnetization(self.lattice_ab, modulations)
        Mx, My = self.uvw2xyz(gamma, self.magnetization_abc[0], self.magnetization_abc[1])
        Mz = self.magnetization_abc[2]
        self.magnetization_xyz = (Mx,My,Mz)

    # Provide these unpacked properties for easy acces: 
    # >>> MagneticLattice.Mx
    #     np.array(...)
    @property
    def Mx(self):
        return self.magnetization_xyz[0]
    
    @property
    def My(self):
        return self.magnetization_xyz[1]
    
    @property
    def Mz(self):
        return self.magnetization_xyz[2]
    
    @property
    def X(self):
        return self.lattice_xy[0]
    
    @property
    def Y(self):
        return self.lattice_xy[1]

    @classmethod
    def uvw2xyz(cls, gamma: float, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''Convert from crystal to cartesian coordinates.

        A and B are matrices of points on the 2D lattice, as from `A, B = np.meshgrid(...)`
        '''
        cg = np.cos(np.radians(gamma))
        sg = np.sin(np.radians(gamma))

        return (A + B*cg, B*sg)

    @classmethod
    def magnetization(cls, lattice: tuple[np.ndarray, np.ndarray], modulations: list[Modulation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Determine magnetization function for arbitrary number of modulations.

        Parameters
        ----------
        lattice: [A, B]
            Lattice points in int units. 
        modulations : list[dict]
            Modulation vectors [{'q': [k, 0, 0], 'ph0':0.0, 'M': [[0,Mxs1],[0,0],[Mzc1,0]]}, ...]

        Returns:
        magnetization: np.ndarray
            Magnetization calculated on the lattice in lattice coordinates. Shape(magnetization) = (shape(A), 3)

        Notes
        -----
        The implemented formulas are in lattice coordinates
        '''
        NA, NB = lattice
        
        Mabc = np.zeros(tuple([3]+list(NA.shape)))

        for mod in modulations:
            ph = NA*mod.qx + NB*mod.qy + mod.phi0

            Mabc += np.einsum('ij,k->kij', np.cos(2*np.pi * ph), mod.Mcos_vec)
            Mabc += np.einsum('ij,k->kij', np.sin(2*np.pi * ph), mod.Msin_vec)

            
        return (Mabc[0], Mabc[1], Mabc[2])
    
def process_id_siteline(site_line: str):
    '''Process restriction on magnetization from ISODISTORT database.
    Copy lin efrom IDODISTORT page, copy results from the console.
    
    Format:
    multiplicity, symbol, (x,y,z;mx,my,mz), (modulations)
    '''

    j, Wsymbol, Wsite, modulation_somponents = site_line.split()

    Mn = modulation_somponents[1:-1].split(';')

    M = [M.replace('{','[').replace('}',']') for M in Mn]

    ret = 'modulations = [\n'
    for m in M:
        ret += f"\t{{'q':[a,0,0], 'ph0':0, 'M':[{m}] }},\n"

    ret += "]"

    return ret

def dict_fill_defaults(d: dict, kwargs: dict) -> dict:
    for key, val in kwargs.items():
        if key not in d:
            d[key] = val

    return d

def plot_arrows(ax: Axes, mlat: MagneticLattice, representation: str='', plot_kwargs: dict={}) -> object:
    print('Plotitng arrows')

    arrow_styles = dict(pivot='middle', scale=80, width=0.003)
    if representation == 'skyrmion':
        Mrho = np.sqrt(mlat.Mx**2 + mlat.My**2)
        Mr = np.sqrt(Mrho**2 + mlat.Mz**2)
        th = np.arctan2(mlat.Mz, Mrho)
        phi = np.arctan2(mlat.My, mlat.Mx)

        # this is skyrmion colormap: (hue, light, saturation)
        phispan = 2*np.pi
        h = np.mod(phi+0, phispan)/phispan
        l = (th+np.pi/2)/(np.pi)
        s = np.ones(h.shape)
        # s = Mr/Mr.max()   # Length encoded in saturation, degenerated for theta=0 or 1
        
        color = np.array([ colorsys.hls_to_rgb(h[it_i,it_j], l[it_i,it_j], s[it_i,it_j])
                        for it_i in range(h.shape[0])
                        for it_j in range(h.shape[1])])

        # set background color as gray
        g = 128/256
        ax.set_facecolor((g,g,g))

        c = color.reshape(-1, color.shape[-1])

        # For in dividual color for each arrow we must provide flattned inputs
        plt_obj = ax.quiver(mlat.X.flatten(), mlat.Y.flatten(), mlat.Mx.flatten(), mlat.My.flatten(), 
                color=c, **arrow_styles)
    elif representation == '2D':
        plt_obj = ax.quiver(mlat.X, mlat.Y, mlat.Mx, mlat.My, **arrow_styles, **plot_kwargs)
    else:
        plt_obj = ax.quiver(mlat.X, mlat.Y, mlat.Mx, mlat.My, mlat.Mz, **arrow_styles, **plot_kwargs)
    
    return plt_obj

def plot_density(ax: Axes, mlat: MagneticLattice, skyrmion_palette: bool=False, plot_kwargs: dict={}) -> object:
    '''Plot density magnetization as pixel colors'''

    print('Plotting density')
    
    plot_defaults = dict(
        vmin=mlat.Mz.min(), vmax=mlat.Mz.max(),
        cmap=cm('Spectral'),
        shading='nearest'
        )
    plot_kwargs = dict_fill_defaults(plot_kwargs, plot_defaults)

    dx = mlat.X[1,1]-mlat.X[0,0]
    dy = mlat.Y[1,1]-mlat.Y[0,0]

    # Calculate pixel colors
    if skyrmion_palette:
        Mrho = np.sqrt(mlat.Mx**2 + mlat.My**2)
        Mr = np.sqrt(Mrho**2 + mlat.Mz**2)
        th = np.arctan2(mlat.Mz, Mrho)
        phi = np.arctan2(mlat.My, mlat.Mx)

        # this is skyrmion colormap: (hue, light, saturation)
        phispan = 2*np.pi
        h = np.mod(phi+0, phispan)/phispan
        l = (th+np.pi/2)/(np.pi)
        s = np.ones(h.shape)
        # s = Mr/Mr.max()   # Length encoded in saturation, degenerated for theta=0 or 1
        
        color = np.array([ colorsys.hls_to_rgb(h[it_i,it_j], l[it_i,it_j], s[it_i,it_j])
                        for it_i in range(h.shape[0])
                        for it_j in range(h.shape[1])])
        

        px_color = color.reshape(list(h.shape)+[3])
    else:
        px_color = mlat.Mz


    # im = ax.imshow(px_color, origin='lower', extent=(mlat.X[0][0]-dx/2, mlat.X[-1][-1]+dx/2, mlat.Y[0][0]*np.sin(np.radians(mlat.gamma))-dy/2, mlat.Y[-1][-1]*np.sin(np.radians(mlat.gamma))+dy/2),
    #                 **plot_kwargs)
    
    im = ax.pcolormesh(mlat.X, mlat.Y, px_color, **plot_kwargs)
    
    return im

def test_interpolations(mag_lattice: MagneticLattice) -> Figure:
    int_types = ['antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']
    plot_mosaic = list(np.reshape(int_types+[str(x) for x in range(2)], (5,4)))
    margins = dict(top=0.975,bottom=0.025,left=0.0,right=1.0,hspace=0.3,wspace=0.0)
    fig, axd = plt.subplot_mosaic(mosaic=plot_mosaic, figsize=(12,11), gridspec_kw=margins)

    for int_type, ax in axd.items():
        if int_type not in int_types:
            ax.remove()
            continue

        ax.set_title(int_type)
        plt_obj = plot_density(ax, mag_lattice, skyrmion_palette=False, plot_kwargs=dict(interpolation=int_type))

    return fig

# 1. Input z ISODISTORTA

def plot():
    # Plot options
    qRangeScale = 1.5   # extension of the plot in units of modulation vector
    N = 64        # density of lattice points/plotted elements
    Narrs = 16

    # site_line = '1 a (0,0,0;0,0,0) ({0,0},{0,Mys1},{Mzc1,0};{0,-Mys1},{0,0},{Mzc1,0}) '
    # print(process_id_siteline(site_line))

    Mxs1, Mzc1 = 3, 3
   
    mssg = MSSG_tables['115.2.68.12.m283.1']
    print('Chosen MSSG:')
    print(mssg)
    modulations = mssg.eval(dict(a=1/N, b=0, A=2, B=2, C=1, D=1))
    print('Evaluated modulations:')
    print(modulations)

    mag_lattice = MagneticLattice(N=N, gamma=mssg.gamma, modulations=modulations)
    mag_lattice_sub = MagneticLattice(N=N, Nsub=N/Narrs, gamma=mssg.gamma, modulations=modulations)

    ##############
    # Plot
    # fig = test_interpolations(mag_lattice)
    fig, ax = plt.subplots(tight_layout=True, figsize=(9,9))

    plt.title(f'Magnetic textures of MSSG={mssg.name}')

    # interesting interpolation schemes: [nearest, Gouraud]
    # plt_obj_dens = plot_density(ax, mag_lattice, skyrmion_palette=False,
    #                        plot_kwargs=dict(shading='Gouraud', cmap=cm('Spectral')))

    arrow_styles = dict(alpha=1, cmap=cm('Spectral'))
    plt_obj_arr = plot_arrows(ax, mag_lattice_sub, representation='3D', plot_kwargs=arrow_styles)
    fig.colorbar(plt_obj_arr, ax=ax, fraction=0.03, label='Mz')

    # plt.axis('square')
    # fig.colorbar()

    RS = 1
    ax.set_xlim(-N/RS, N/RS)
    ax.set_ylim(-N/RS, N/RS)


    return fig



if __name__ == '__main__':
    print(MSSG_tables['191.2.83.7.m236.1'])

    # fig = plot()
    # fig.savefig('skyrmion.png', dpi=100)
    # fig.savefig('skyrmion.pdf')