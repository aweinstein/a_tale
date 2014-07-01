import itertools

import networkx as nx
import numpy as np
from numpy.linalg import eig, eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

def save_fig(fig, file_name):
    if isinstance(fig, list):
        pdf = PdfPages(file_name)
        for f in fig:
            pdf.savefig(f)
        pdf.close()
    else:
        fig.savefig(file_name, format='pdf', dpi=300)
    print 'File %s created' % file_name

def make_grid(m, n, offset=(0,0)):
    """Make a graph representing a grid of m rows and n columns.

    Note: This is a simplified version fo nx.grid_2d_graph
    """
    G = nx.Graph()
    r_offset, c_offset = offset
    rows = range(r_offset, r_offset + m)
    cols = range(c_offset, c_offset + n)
    nodes = list(itertools.product(rows, cols)) # The nodes are in row-major
                                                # order
    G.add_nodes_from(nodes)

    G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in cols
                      if i>r_offset )
    G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in cols
                      if j>c_offset )

    return G, nodes

def make_rooms(n_rooms, m, n):
    """Make a graph representation of the connection of n_rooms rooms.

    Each room is a m by n grid. The rooms are connected horizontally through
    the middle of the right column.
    """
    # I only use the graphs. I rebuild the nodes list again.
    # I don't know if using the previously generated nodes is better
    rooms = [make_grid(m, n, (0, i*n))[0] for i in range(n_rooms)]
    nodes = list(itertools.product(range(m), range(n_rooms*n)))

    G = reduce(nx.union, rooms)

    # Add edges betwee the rooms
    m_edge = m / 2 # All the edges are in the middle of the column
    for i in range(n_rooms-1):
        G.add_edge( (m_edge, n * i + 1), (m_edge, n * i + 2) )

    return G, nodes


def room_test(m=16, n=20, n_rooms=3, interactive=True):
    G, nodes = make_rooms(n_rooms, m, n)
    L = nx.linalg.laplacian_matrix(G, nodes)
    evalues, evec = eigh(L)
    idxs = np.argsort(evalues)

    b = lambda i: np.reshape(evec[:, idxs[i]], (m, -1))

    # Plotting
    x = np.arange(n * n_rooms)
    y = np.arange(m)
    X,Y = np.meshgrid(x,y)
    fig_1 = plt.figure(1)
    #fig_2 = plt.figure(2)
    ax_3d_1 = Axes3D(fig_1)
    #ax_3d_2 = Axes3D(fig_2)
    colors = 'rbgcy'
    n_plots = 3
    for i in range(n_plots):
        ax_3d_1.plot_surface(X, Y, b(i+1), cstride=1, rstride=1,
                             color=colors[i], alpha=0.3)

    plt.figure(1)
    plt.xlabel('columns')
    plt.ylabel('rows')

    if interactive:
        plt.show()
    else:
        save_fig(fig_1, 'room.pdf')

if __name__ == '__main__':
    room_test(16, 20, 3, interactive=False)
