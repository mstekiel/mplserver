import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import numpy as np
import numpy.typing as npt
import scipy

def magnetization(lattice, modulations):
    '''Determine magnetization function for arbitrary number (n) of modulations.

    Parameters
    ----------
    lattice: (N,N,2)
        Lattice points
    ks : (3,n)
        Modulation vectors
    Ms: (3,2,n)
        Components of the magnetization
    ph0s: (n)
        Phase shifts

    Notes
    -----
    The implemented formulas are surely xyz, or should they be abc?
    '''
    NA, NB = lattice
    
    Mx, My, Mz = np.zeros(tuple([3]+list(NA.shape)))

    for mod in modulations:
        q = mod['q']
        M = np.array(mod['M'])
        ph0 = mod['ph0']
        ph = NA*q[0] + NB*q[1] + ph0

        Mx += M[0,0] * np.cos(2*np.pi * ph) + M[0,1] * np.sin(2*np.pi * ph)
        My += M[1,0] * np.cos(2*np.pi * ph) + M[1,1] * np.sin(2*np.pi * ph)
        Mz += M[2,0] * np.cos(2*np.pi * ph) + M[2,1] * np.sin(2*np.pi * ph)
        
    return Mx, My, Mz

def process_id_siteline(site_line: str):
    '''Process restriction on magnetization from ISODISTORT database.
    Copy lin efrom IDODISTORT page, copy results from the console.
    
    Format:
    multiplicity, symbol, (x,y,z;mx,my,mz), (modulations)
    '''

    j, Wsymbol, Wsite, modulation_somponents = site_line.split()

    Mn = modulation_somponents[1:-1].split(';')

    M = [M.replace('{','[').replace('}',']') for M in Mn]

    return M

def archive():
    # Case1 -> reproduce slides
    # 89.2.63.9.m87.1  P422.1(a,0,0)0s0(0,a,0)000
    gamma = 90
    site_line = '1 a (0,0,0;0,0,0) ({0,0},{0,Mys1},{Mzc1,0};{0,-Mys1},{0,0},{Mzc1,0}) '
    print(process_id_siteline(site_line))

    k = 0.1
    Mys1, Mzc1 = 3, 1
    modulations = [
        {'q': [k, 0, 0], 'ph0':0.0, 'M': [[0,0],[0,Mys1],[Mzc1,0]]},
        {'q': [0,k,0], 'ph0':0.0, 'M': [[0,-Mys1],[0,0],[Mzc1,0]]},
    ]


    # Case2 -> weird elliptical?
    # 177.2.82.5.m149.1 P 622.1(α, 0, 0)000(α, α, 0)000
    gamma = 120
    site_line = ' 1 a (0,0,0;0,0,0) ({0,Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,-Mxs1},{0,0}) '
    print(process_id_siteline(site_line))

    k = 0.1
    Mxs1 = 1
    modulations = [
        {'q': [k, 0, 0], 'ph0':0.0, 'M': [[0,Mxs1],[0,0.50000*Mxs1],[0,0]]},
        {'q': [-k, k, 0], 'ph0':0.0, 'M': [[0,-0.50000*Mxs1],[0,0.50000*Mxs1],[0,0]]},
        {'q': [0,-k,0], 'ph0':0.0, 'M': [[0,-0.50000*Mxs1],[0,-Mxs1],[0,0]]},
    ]


def plot():

    Na = Nb = 15
    gamma = 120

    # Parameters in crystal coordinates

    # Case3 -> yambe2021skyrmion Eq. 7
    # 177.2.82.5.m149.1 P 622.1(α, 0, 0)000(α, α, 0)000
    site_line = ' 1 a (0,0,0;0,0,0) ({0,Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,-Mxs1},{0,0}) '
    print(process_id_siteline(site_line))

    a = 0.1
    Mxs1, Mzc1= 1, 2
    ph = 0.25 # try 0.25 as well
    st = np.sqrt(3)/2
    modulations = [
        {'q': [a, 0, 0], 'ph0':ph, 'M': [[0,0],[0,Mxs1],[Mzc1,0]]},
        {'q': [-a, a, 0], 'ph0':ph, 'M': [[0,-st*Mxs1],[0,-0.5*Mxs1],[Mzc1,0]]},
        {'q': [0,-a,0], 'ph0':ph, 'M': [[0,st*Mxs1],[0,-0.5*Mxs1],[Mzc1,0]]},
    ]

    # Larger scale = smaller arrows
    arrow_scale = 8e+1

    # Calculations to cartesian system
    cg = np.cos(np.radians(gamma))
    sg = np.sin(np.radians(gamma))
    na = np.arange(-Na,Na+1)
    nb = np.arange(-Nb,Nb+1)
    NA, NB = np.meshgrid(na, nb)
    X = NA + NB*cg
    Y = NB*sg

    Mx, My, Mz = magnetization((NA,NB), modulations)
    
    # Plot
    fig, ax = plt.subplots(tight_layout=True)
    ax.quiver(X, Y, Mx, My, Mz, cmap=cm.jet, pivot='middle', scale=8e+1)

    lim_min, lim_max = np.min([X,Y]), np.max([X,Y])
    plt.axis('square')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    print(fig.get_size_inches())

    plt.title('Magnetic textures')

    return fig

if __name__ == '__main__':
    fig = plot()
    fig.savefig('test.png')
    fig.savefig('test.pdf')