import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import colorsys


import numpy as np
import numpy.typing as npt
import scipy

def uvw2xyz(gamma, A, B):
    '''Convert from crystal to cartesian coordinates.

    A and B are matrices of points on the 2D lattice, as from `A, B = np.meshgrid(...)`
    '''
    cg = np.cos(np.radians(gamma))
    sg = np.sin(np.radians(gamma))

    return (A + B*cg, B*sg)

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
    
    moments = np.zeros(tuple([3]+list(NA.shape)))

    for mod in modulations:
        q = mod['q']
        M = np.array(mod['M'], dtype=float)
        ph0 = mod['ph0']
        ph = NA*q[0] + NB*q[1] + ph0

        moments += np.einsum('ij,k->kij', np.cos(2*np.pi * ph), M[:,0])
        moments += np.einsum('ij,k->kij', np.sin(2*np.pi * ph), M[:,1])
        
    return moments

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
    Mxs1 = 1.0
    modulations = [
        {'q': [k, 0, 0], 'ph0':0.0, 'M': [[0,Mxs1],[0,0.50000*Mxs1],[0,0]]},
        {'q': [-k, k, 0], 'ph0':0.0, 'M': [[0,-0.50000*Mxs1],[0,0.50000*Mxs1],[0,0]]},
        {'q': [0,-k,0], 'ph0':0.0, 'M': [[0,-0.50000*Mxs1],[0,-Mxs1],[0,0]]},
    ]



    # Case3 -> yambe2021skyrmion Eq. 7
    # 177.2.82.5.m149.1 P 622.1(α, 0, 0)000(α, α, 0)000
    gamma = 120
    site_line = ' 1 a (0,0,0;0,0,0) ({0,Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,0.50000*Mxs1},{0,0};{0,-0.50000*Mxs1},{0,-Mxs1},{0,0}) '
    print(process_id_siteline(site_line))

    a = 1/16
    Mxs1, Mzc1= 1, 2
    ph = 0. # try 0.25 as well
    st = np.sqrt(3)/2
    modulations = [
        {'q': [a, 0, 0], 'ph0':ph, 'M': [[0,0],[0,Mxs1],[Mzc1,0]]},
        {'q': [-a, a, 0], 'ph0':ph, 'M': [[0,-st*Mxs1],[0,-0.5*Mxs1],[Mzc1,0]]},
        {'q': [0,-a,0], 'ph0':ph, 'M': [[0,st*Mxs1],[0,-0.5*Mxs1],[Mzc1,0]]},
    ]

        # Case4
    # Magnetic superspace group: 175.2.80.1.m140.1  P6/m'(a,b,0)00(-a-b,a,0)00
    gamma = 120
    site_line = ' 1 a (0,0,z;0,0,mz) ({0,Mxs1},{0,Mys1},{Mzc1,0};{0,-Mys1},{0,Mxs1-Mys1},{Mzc1,0};{0,-Mxs1+Mys1},{0,-Mxs1},{Mzc1,0}) '
    print(process_id_siteline(site_line))

    a, b = 3/Na, 0
    Mxs1, Mys1 = 0, 1
    Mzc1 = 0.5
    modulations = [
        {'q':[a,b,0], 'ph0':0, 'M':[[0,Mxs1],[0,Mys1],[Mzc1,0]] },
        {'q':[-a-b,a,0], 'ph0':0, 'M':[[0,-Mys1],[0,Mxs1-Mys1],[Mzc1,0]] },
        {'q':[b,-a-b,0], 'ph0':0, 'M':[[0,-Mxs1+Mys1],[0,-Mxs1],[Mzc1,0]] },
    ]

# 1. Input z ISODISTORTA
# 2. pcolormesh na ciagly rozklad

def plot():
    # Plot options
    Na = Nb = 100
    arrow_scale = 8e+1

    # Case5 -> reproduce pdf card
    # 89.2.63.9.m87.1  P422.1(a,0,0)0s0(0,a,0)000
    gamma = 90
    site_line = '1 a (0,0,0;0,0,0) ({0,0},{0,Mys1},{Mzc1,0};{0,-Mys1},{0,0},{Mzc1,0}) '
    print(process_id_siteline(site_line))

    k = 1.5/Na
    Mxs1, Mzc1 = 3, 3
    modulations = [
        {'q': [k, 0, 0], 'ph0':0.0, 'M': [[0,Mxs1],[0,0],[Mzc1,0]]},
        {'q': [0,k,0], 'ph0':0.0, 'M': [[0,0],[0,Mxs1],[Mzc1,0]]},
    ]




    ##############
    # Calculations
    na = np.arange(-Na,Na+1)
    nb = np.arange(-Nb,Nb+1)
    A, B = np.meshgrid(na, nb)
    X, Y = uvw2xyz(gamma, A, B)

    Ma, Mb, Mc = magnetization((A,B), modulations)
    Mx, My = uvw2xyz(gamma, Ma, Mb)
    Mz = Mc

    # Plot
    fig, ax = plt.subplots(tight_layout=True)
    # ax.quiver(X, Y, Mx, My, Mz, cmap=cm.jet, pivot='middle', scale=8e+1)

    Mrho = np.sqrt(Mx**2 + My**2)
    Mr = np.sqrt(Mrho**2 + Mz**2)
    th = np.arctan2(Mz, Mrho)
    phi = np.arctan2(My, Mx)

    phispan = 2*np.pi
    h = np.mod(phi+0, phispan)/phispan
    l = (th+np.pi/2)/(np.pi)
    s = np.ones(h.shape)
    # s = Mr/Mr.max()   # Length encoded in saturation, degenerated for theta=0 or 1
    
    color = np.zeros((h.shape[0],h.shape[1],3))
    for it_i in range(h.shape[0]):
        for it_j in range(h.shape[1]):
            color[it_i,  it_j] = colorsys.hls_to_rgb(h[it_i,it_j], l[it_i,it_j], s[it_i,it_j])
    # hsl = np.array([h,np.ones(shape=h.shape),l]).T
    # color = colorsys.hls_to_rgb(h,l,s)
    # cl = np.vectorize(colorsys.hls_to_rgb)
    # color = cl([h,l,s])

    g = 128/256
    ax.set_facecolor((g,g,g))

    cp, sp = (np.cos(phi)+1)/2, (np.sin(phi)+1)/2
    ct, st = (np.cos(th)+1)/2, (np.sin(th)+1)/2
    ptd = np.sqrt(3)/2
    # r = 0.5*np.einsum('i,ikl->kl', [0, -1], [np.cos(phi), np.sin(phi)])+0.5
    # g = 0.5*np.einsum('i,ikl->kl', [0.5, ptd], [np.cos(phi), np.sin(phi)])+0.5
    # b = 0.5*np.einsum('i,ikl->kl', [-0.5, ptd], [np.cos(phi), np.sin(phi)])+0.5
    # color = np.array([r,g,b]).T

    Mx = np.zeros(X.shape)
    My = np.zeros(X.shape)

    c = color.reshape(-1, color.shape[-1])
    ax.quiver(X.flatten(), Y.flatten(), Mx.flatten(), My.flatten(), 
              color=c, pivot='middle', scale=8e+1, width=0.008)

    lim_min, lim_max = np.min([X,Y]), np.max([X,Y])
    lcut = 1
    plt.axis('square')
    ax.set_xlim(lim_min/lcut, lim_max/lcut)
    ax.set_ylim(lim_min/lcut, lim_max/lcut)

    plt.title('Magnetic textures')

    return fig

if __name__ == '__main__':
    fig = plot()
    fig.savefig('skyrmion.png')
    fig.savefig('skyrmion.pdf')