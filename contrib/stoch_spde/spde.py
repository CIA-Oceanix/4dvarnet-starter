from contrib.stoch_spde.scipy_sparse_tools import *
import torch.nn.functional as F

def slice_sparse_Tensor(A,start,stop,new_size,by=0):

    m, n = A.size()
    index = A.coalesce().indices()
    value = A.coalesce().values()
    sel = (index[by]>=start) & (index[by]<stop)
    sel = torch.squeeze(sel.nonzero())
    row = index[0][sel]
    col = index[1][sel]
    if by==0:
        row = row-start
    else:
        col=col-start
    val = value[sel]
    if by==0:
        m, n = new_size, n
    else:
        m, n = m, new_size
    B = torch.sparse_coo_tensor(torch.LongTensor([row.tolist(),col.tolist()]),
                                         torch.FloatTensor(val.tolist()),
                                         torch.Size([m,n])).coalesce().to(device)  
    return B
    
def pow_diff_operator(A,pow,sparse=False,split=False,n_t=None):
    B = A
    m = A.size()[0]
    n = A.size()[1]
    for i in range(pow-1):
        if sparse==False:
            B = torch.matmul(B,A)
        else:
            if not split:
                #B = torch.sparse.mm(B,A).coalesce()
                if (i % 2) != 0:
                    B = torch.sparse.mm(B,A.t()).coalesce()
                else:
                    B = torch.sparse.mm(B,A).coalesce()
            else:
                # define sparse observation operator
                row = []
                col = []
                val = [] 
                for k in range(n_t):
                    A_sl = slice_sparse_Tensor(A,k*(n//n_t),(k+1)*(n//n_t),new_size=n//n_t,by=1)
                    Bk = torch.sparse.mm(B,A_sl).coalesce()
                    indexA = Bk.indices()
                    valueA = Bk.values()
                    row.extend(indexA[0].tolist())
                    col.extend(( ((n//n_t)*k) + indexA[1]).tolist())
                    val.extend(valueA.tolist())
                B = torch.sparse_coo_tensor(torch.LongTensor([row,
                                                              col]),
                                         torch.FloatTensor(val),
                                         torch.Size([m,n])).coalesce().to(device)            
    #B=(1./2)*(B.t()+B)
    return B

def get_boundary_index(Nx,Ny):
    # bottom
    border = np.arange(0,Nx).tolist()
    # top
    border.extend(np.arange(Nx*(Ny-1),Nx*Ny).tolist())
    # left
    border.extend(np.arange(Nx,Nx*(Ny-1),Nx).tolist())
    # right
    border.extend(np.arange(2*Nx-1,Nx*(Ny-1),Nx).tolist())
    # sort border
    border.sort()
    border = np.array(border)
    return border
    #return torch.from_numpy(border).to(device)

def Gamma(tensor):
    return(torch.tensor(tensor,dtype=torch.float).lgamma().exp())

def regularize_variance(nu, kappa):
    # nu is an integer
    # kappa is a nbNodes-tensor
    d = 2
    pi = torch.acos(torch.zeros(1)).item() * 2
    return( ((Gamma(nu+d/2)*((4*pi)**(d/2))*(kappa**(2*nu))) / Gamma(nu) )**(1/2) )

def init_BS_basis(Nx,Ny,Nt,Nfbx,Nfby,Nfbt):
        # 3D B-splines basis functions
        bspl_basis_alongX = BSpline(domain_range=[-1,Nx+1], n_basis=Nfbx, order=3)
        bspl_basis_alongY = BSpline(domain_range=[-1,Ny+1], n_basis=Nfby, order=3)
        # time domain of definition: [t-Ntdt,t]
        bspl_basis_alongT = BSpline(domain_range=[-1*Nt,1], n_basis=Nfbt, order=3)
        # compute 3D B-splines basis functions
        grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(Nx),
                                                     torch.arange(Ny),
                                                     torch.arange(Nt)])),
                                              (3,Nx*Ny*Nt))
        bX=torch.from_numpy(bspl_basis_alongX(grid[0])[:,:,-1])
        bY=torch.from_numpy(bspl_basis_alongY(grid[1])[:,:,-1])
        bT=torch.from_numpy(bspl_basis_alongT(grid[2])[:,:,-1])
        bXY= torch.stack([ torch.einsum('i,j->ij',
                       bX[:,i],
                       bY[:,i]).reshape(bX.shape[0]*bY.shape[0]) for i in range(bX.shape[1]) ])
        bXY = torch.transpose(bXY,0,1)
        bXYT= torch.stack([ torch.einsum('i,j->ij',
                       bXY[:,i],
                       bT[:,i]).reshape(bXY.shape[0]*bT.shape[0]) for i in range(bXY.shape[1]) ])
        bXY = torch.transpose(bXY,0,1)
        return bXYT.to(device)

def DiffOperator_FUDM3(Nx, Ny, dx, dy, m, H, kappa, stabilization = True):
# kappa is 1*(Ny*Nx)
# m is 2*(Ny*Nx)
# H is 2*2*(Ny*Nx)

    nb_nodes = Nx * Ny
    nodes = torch.arange(0,nb_nodes).to(device)
    liste_k = []

    #########################################
    # Voisins i-1 / i+1 -> dx discretization
    #########################################

    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(nodes+1,Nx) != 0)[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    k1 = torch.stack((indices_neighbours.float(), indices_neighbours.float() + 1,
                          -1 * (H[0, 0, indices_neighbours])/(dx**2)))
    if m is not None:
        if stabilization==False:
            k1[2] = k1[2] +  m[0, indices_neighbours]/(2*dx)
        if stabilization==True:
            N = len(indices_neighbours)
            a1p = torch.max(m[0, indices_neighbours],torch.zeros(N).to(device))
            a1m = torch.min(m[0, indices_neighbours],torch.zeros(N).to(device))
            k1[2] = k1[2] + (2*a1p + 6*a1m)/(6*dx)
        liste_k.append(k1)

    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(nodes+1,Nx) != 1)[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    k2 = torch.stack((indices_neighbours.float(), indices_neighbours.float() - 1,
                          -1 * (H[0, 0, indices_neighbours])/(dx**2)))
    if m is not None:
        if stabilization==False:
            k2[2] = k2[2] - m[0, indices_neighbours]/(2*dx)
        if stabilization==True:
            N = len(indices_neighbours)
            a1p = torch.max(m[0, indices_neighbours],torch.zeros(N).to(device))
            a1m = torch.min(m[0, indices_neighbours],torch.zeros(N).to(device))
            k2[2] = k2[2] - (6*a1p + 2*a1m)/(6*dx)
        liste_k.append(k2)

    #########################################
    # Voisins i-2 / i+2 -> dx discretization
    #########################################
    
    ## Voisin à droite (i+2) Points de la grille concernés : Retrait du bord droit
    index = torch.where( (torch.fmod(nodes+1,Nx) != 0) & (torch.fmod(nodes+1,Nx) != (Nx-1)) )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    if m is not None:
        if stabilization==True:
            N = len(indices_neighbours)
            a1p = torch.max(m[0, indices_neighbours],torch.zeros(N).to(device))
            a1m = torch.min(m[0, indices_neighbours],torch.zeros(N).to(device))
            k3 = torch.stack((indices_neighbours.float(),
                               indices_neighbours.float() + 2,
                               -1*a1m/(6*dx)))
            liste_k.append(k3)

    ## Voisin à gauche (i-2) Points de la grille concernés : Retrait du bord gauche
    index = torch.where( (torch.fmod(nodes+1,Nx) != 1) & (torch.fmod(nodes+1,Nx) != 2) )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    if m is not None:
        if stabilization==True:
            N = len(indices_neighbours)
            a1p = torch.max(m[0, indices_neighbours],torch.zeros(N).to(device))
            a1m = torch.min(m[0, indices_neighbours],torch.zeros(N).to(device))
            k4 = torch.stack((indices_neighbours.float(),
                               indices_neighbours.float() - 2,
                               a1p/(6*dx)))
            liste_k.append(k4)
    
    #########################################
    # Voisins j-1 / j+1 -> dy discretization
    #########################################

    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((nodes+1) <= (Ny-1)*Nx )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    k5 = torch.stack((indices_neighbours.float(), indices_neighbours.float() + Nx,
                         -1 * (H[1, 1, indices_neighbours])/(dy**2)))
    if m is not None:
        if stabilization==False:
            k5[2] = k5[2] + m[1, indices_neighbours]/(2*dy)
        if stabilization==True:
            N = len(indices_neighbours)
            a2p = torch.max(m[1, indices_neighbours],torch.zeros(N).to(device))
            a2m = torch.min(m[1, indices_neighbours],torch.zeros(N).to(device))
            k5[2] = k5[2] + (2*a2p + 6*a2m)/(6*dy)
        liste_k.append(k5)

    ## Voisin du bas points de la grille concernés : Retrait du bord bas
    index = torch.where((nodes+1) >= (Nx+1) )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    k6 = torch.stack((indices_neighbours.float(), indices_neighbours.float() - Nx,
                          -1 * (H[1, 1, indices_neighbours])/(dy**2)))
    if m is not None:
        if stabilization==False:
            k6[2] = k6[2] - m[1, indices_neighbours]/(2*dy)
        if stabilization==True:
            N = len(indices_neighbours)
            a2p = torch.max(m[1, indices_neighbours],torch.zeros(N).to(device))
            a2m = torch.min(m[1, indices_neighbours],torch.zeros(N).to(device))
            k6[2] = k6[2] - (6*a2p + 2*a2m)/(6*dy)
        liste_k.append(k6)

    #########################################
    # Voisins j-2 / j+2 -> dy discretization
    #########################################
    
    ## Voisin à droite (j+2) Points de la grille concernés : Retrait du bord haut
    index = torch.where((nodes+1) <= (Ny-2)*Nx )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    if m is not None:
        if stabilization==True:
            N = len(indices_neighbours)
            a2p = torch.max(m[1, indices_neighbours],torch.zeros(N).to(device))
            a2m = torch.min(m[1, indices_neighbours],torch.zeros(N).to(device))
            k7 = torch.stack((indices_neighbours.float(),
                               indices_neighbours.float() + 2*Nx,
                               -1*a2m/(6*dy)))
            liste_k.append(k7)

    ## Voisin à gauche (j-2) Points de la grille concernés : Retrait du bord bas
    index = torch.where((nodes+1) >= (2*Nx+1) )[0]
    indices_neighbours = torch.index_select(nodes,0,index)
    if m is not None:
        if stabilization==True:
            N = len(indices_neighbours)
            a2p = torch.max(m[1, indices_neighbours],torch.zeros(N).to(device))
            a2m = torch.min(m[1, indices_neighbours],torch.zeros(N).to(device))
            k8 = torch.stack((indices_neighbours.float(),
                               indices_neighbours.float() -2*Nx,
                               a2p/(6*dx)))
            liste_k.append(k8)

    #####################
    # Point central i,j #
    #####################

    if torch.is_tensor(kappa):
        k9 = torch.stack((nodes.float(), nodes.float(),
                              (kappa[0,nodes]**2)))
    else:
        k9 = torch.stack((nodes.float(), nodes.float(), 
                              kappa**2))
                                  
    if H is not None:
        k9[2] = k9[2] + 2 * (H[0, 0, nodes]/(dx**2) + H[1, 1, nodes]/(dy**2))
    else:
        k9[2] = k9[2] + 2 * (1./(dx**2) + 1./(dy**2))*torch.ones(len(nodes)).to(device)
        

    if ( (m is not None) and (stabilization==True) ):
        N = len(nodes)
        a1p = torch.max(m[0, nodes],torch.zeros(N).to(device))
        a1m = torch.min(m[0, nodes],torch.zeros(N).to(device))
        a2p = torch.max(m[1, nodes],torch.zeros(N).to(device))
        a2m = torch.min(m[1, nodes],torch.zeros(N).to(device))
        k9[2] = k9[2] + 3*(a1p-a1m)/(6*dx) + 3*(a2p-a2m)/(6*dy)
    liste_k.append(k9)

    ################################################
    # Voisins i-1,j-1/ i-1,j+1 / i+1,j-1 / i+1/j+1 #
    #           -> dxdy discretization             #
    ################################################

    if H is not None:
        ## Voisin en haut à droite Points de la grille concernés : Retrait du bord haut et droit
        index = torch.where( (torch.fmod(nodes+1,Nx) != 0) & ((nodes+1)<= (Ny-1)*Nx) )[0]
        indices_neighbours = torch.index_select(nodes,0,index)
        k10 = torch.stack((indices_neighbours.float(), indices_neighbours.float() + Nx+1,
                         (H[0, 1, indices_neighbours] + H[1, 0, indices_neighbours])/(2*dx*dy)))
        liste_k.append(k10)
        ## Voisin en haut à gauche Points de la grille concernés : Retrait du bord haut et gauche
        index = torch.where( (torch.fmod(nodes+1,Nx) != 1) & ((nodes+1)<= (Ny-1)*Nx) )[0]
        indices_neighbours = torch.index_select(nodes,0,index)
        k11 = torch.stack((indices_neighbours.float(), indices_neighbours.float() + Nx-1, -1*(H[0, 1, indices_neighbours] + H[1, 0,indices_neighbours])/(2*dx*dy)))
        liste_k.append(k11)
        ## Voisin en bas à droite Points de la grille concernés : Retrait du bord bas et droit
        index = torch.where( (torch.fmod(nodes+1,Nx) != 0) & ((nodes+1)>=(Nx+1)) )[0]
        indices_neighbours = torch.index_select(nodes,0,index)
        k12 = torch.stack((indices_neighbours.float(), indices_neighbours.float() - Nx+1, -1*(H[0, 1, indices_neighbours] + H[1, 0,indices_neighbours])/(2*dx*dy)))
        liste_k.append(k12)
        ## Voisin en bas à gauche Points de la grille concernés : Retrait du bord bas et gauche
        index = torch.where( (torch.fmod(nodes+1,Nx) != 1) & ((nodes+1)>=(Nx+1)) )[0]
        indices_neighbours = torch.index_select(nodes,0,index)
        k13 = torch.stack((indices_neighbours.float(), indices_neighbours.float() - Nx-1, (H[0, 1, indices_neighbours] + H[1, 0,indices_neighbours])/(2*dx*dy)))
        liste_k.append(k13)

    k = torch.cat(liste_k,dim=1)
    #torch.sparse_coo_tensorres = torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nb_nodes,nb_nodes])).to(device)
    res = torch.sparse_coo_tensor(k[0:2].long(), k[2], torch.Size([nb_nodes,nb_nodes]),
                                 requires_grad=True).coalesce().to(device)
    return res

def DiffOperator_FUDM1(Nx, Ny, dx, dy, m, H, kappa, stabilization = False):
# kappa is 1*(Ny*Nx)
# m is 2*(Ny*Nx)
# H is 2*2*(Ny*Nx)

    nbNodes = Nx * Ny 
    indices = torch.arange(0,nbNodes).to(device)
    
    ############################################
    # Voisins gauche/droite -> dx discretization
    ############################################
    
    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(indices+1,Nx) != 0)[0]
    indicesVoisins = torch.index_select(indices,0,index)    
    k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))                   
    if m is not None:
        if stabilization==False:
            k1[2] = k1[2] +  m[0, indicesVoisins]/(2*dx)
        if stabilization==True:
            k1[2] = k1[2] +  F.relu(-1.*torch.sign(m[0, indicesVoisins]))*m[0, indicesVoisins]/(dx)
                                      
    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(indices+1,Nx) != 1)[0]
    indicesVoisins = torch.index_select(indices,0,index)    
    k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))
    if m is not None:
        if stabilization==False:
            k2[2] = k2[2] - m[0, indicesVoisins]/(2*dx)
        if stabilization==True:
            k2[2] = k2[2] -  F.relu(torch.sign(m[0, indicesVoisins]))*m[0, indicesVoisins]/(dx)
                                      
    ############################################
    # Voisins haut/bas -> dy discretization
    ############################################
    
    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((indices+1) <= (Ny-1)*Nx )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                         -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    if m is not None:
        if stabilization==False:
            k3[2] = k3[2] + m[1, indicesVoisins]/(2*dy)
        if stabilization==True:
            k3[2] = k3[2] +  F.relu(-1.*torch.sign(m[1, indicesVoisins]))*m[1, indicesVoisins]/(dy)
                                      
    ## Voisin du bas Points de la grille concernés : Retrait du bord bas
    index = torch.where((indices+1) >= (Nx+1) )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx,
                          -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    if m is not None:
        if stabilization==False:
            k4[2] = k4[2] - m[1, indicesVoisins]/(2*dy)
        if stabilization==True:
            k4[2] = k4[2] - F.relu(torch.sign(m[1, indicesVoisins]))*m[1, indicesVoisins]/(dy)
            
    ############################################
    # Point central
    ############################################

    if torch.is_tensor(kappa):
        k5 = torch.stack((indices.float(), indices.float(),
                              (kappa[0,indices]**2)))
    else:
        k5 = torch.stack((indices.float(), indices.float(), 
                              kappa**2))
                                  
    if H is not None:
        k5[2] = k5[2] + 2 * (H[0, 0, indices]/(dx**2) + H[1, 1, indices]/(dy**2))
    else:
        k5[2] = k5[2] + 2 * (1./(dx**2) + 1./(dy**2))*torch.ones(len(indices)).to(device)
        
    if ( (m is not None) and (stabilization==True) ):        
        k5[2] = k5[2] + torch.sign(m[0,indices])*m[0,indices]/(dx) +\
                        torch.sign(m[1,indices])*m[1,indices]/(dy)
        
    #################################################################
    # Voisins haut/bas gauche/ haut/bas droite -> dxdy discretization
    #################################################################
    
    if H is not None: 
        ## Voisin en haut à droite Points de la grille concernés : Retrait du bord haut et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k6 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx+1,
                         (H[0, 1, indicesVoisins] + H[1, 0, indicesVoisins])/(2*dx*dy)))
        ## Voisin en haut à gauche Points de la grille concernés : Retrait du bord haut et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k7 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx-1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(2*dx*dy)))
        ## Voisin en bas à droite Points de la grille concernés : Retrait du bord bas et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k8 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx+1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(2*dx*dy)))
        ## Voisin en bas à gauche Points de la grille concernés : Retrait du bord bas et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k9 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx-1, (H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(2*dx*dy)))
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5, k6, k7, k8, k9),dim=1)
    else:
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5),dim=1)
        
    res = torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes])).to(device) 
    return res

def DiffOperator_FDM(Nx, Ny, dx, dy, m, H, kappa):
# kappa is 1*(Ny*Nx)
# m is 2*(Ny*Nx)
# H is 2*2*(Ny*Nx)

    nbNodes = Nx * Ny 
    indices = torch.arange(0,nbNodes).to(device)
    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(indices+1,Nx) != 0)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2) + m[0, indicesVoisins]/dx))
    elif ( (m is None) and (H is not None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))
    elif ( (m is not None) and (H is None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * m[0, indicesVoisins]/dx))
    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(indices+1,Nx) != 1)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2) - m[0, indicesVoisins]/dx))
    elif ( (m is None) and (H is not None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))
    elif ( (m is not None) and (H is None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          m[0, indicesVoisins]/dx))
    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((indices+1) <= (Ny-1)*Nx )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                          -1 * (H[1, 1, indicesVoisins])/(dy**2) + m[1, indicesVoisins]/dy))
    elif ( (m is None) and (H is not None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                         -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    elif ( (m is not None) and (H is None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                          -1 * m[1, indicesVoisins]/dy))
    ## Voisin du bas Points de la grille concernés : Retrait du bord bas
    index = torch.where((indices+1) >= (Nx+1) )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, 
                          -1 * (H[1, 1, indicesVoisins])/(dy**2) - m[1, indicesVoisins]/dy))
    elif ( (m is None) and (H is not None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx,
                          -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    elif ( (m is not None) and (H is None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, 
                          m[1, indicesVoisins]/dy))
    ## Point central
    if H is not None:
        if torch.is_tensor(kappa):
            k5 = torch.stack((indices.float(), indices.float(),
                             kappa[0,indices]**2 + 2 * (H[0, 0, indices]/(dx**2) + H[1, 1, indices]/(dy**2))))
        else:
            k5 = torch.stack((indices.float(), indices.float(),
                            kappa**2 + 2 * (H[0, 0, indices]/(dx**2) + H[1, 1, indices]/(dy**2))))
    else:
        if torch.is_tensor(kappa):
            k5 = torch.stack((indices.float(), indices.float(),
                              (kappa[0,indices]**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device )))
        else:
            k5 = torch.stack((indices.float(), indices.float(), 
                              (kappa**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device) ))
    if H is not None: 
        ## Voisin en haut à droite Points de la grille concernés : Retrait du bord haut et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k6 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx+1,
                         (H[0, 1, indicesVoisins] + H[1, 0, indicesVoisins])/(4*dx*dy)))
        ## Voisin en haut à gauche Points de la grille concernés : Retrait du bord haut et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k7 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx-1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Voisin en bas à droite Points de la grille concernés : Retrait du bord bas et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k8 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx+1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Voisin en bas à gauche Points de la grille concernés : Retrait du bord bas et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k9 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx-1, (H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5, k6, k7, k8, k9),dim=1)
    else:
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5),dim=1)
        
    res = torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes])).to(device) 
    return res

def DiffOperator_Isotropic(Nx, Ny, dx, dy, kappa):
# kappa is a scalar

    nbNodes = Nx * Ny
    indices = torch.arange(0,nbNodes).to(device)
    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(indices+1,Nx) != 0)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1, (-1./(dx**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(indices+1,Nx) != 1)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1, (-1./(dx**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((indices+1) <= (Ny-1)*Nx )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx, (-1./(dy**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin du bas Points de la grille concernés : Retrait du bord bas
    index = torch.where((indices+1) >= (Nx+1) )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, (-1/(dy**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Point central
    k5 = torch.stack((indices.float(), indices.float(), (kappa**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device) ))
    ## Tous les voisins
    k = torch.cat((k1, k2, k3, k4, k5),dim=1)
    return(torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes])))

