#Bogdan Bintu
#Copyright Presidents and Fellows of Harvard College, 2017.

import sys,os,glob
import numpy as np
from scipy import spatial
import pickle as pickle
import matplotlib.pylab as plt

## General tools
def offdiag_elems(mat):
    vals=[]
    for i in range(mat.shape[0]):
        for j in range(i):
            vals.append(mat[i,j])
    return np.array(vals)
def cor_fig(im1,im2,title=None,xlim=None,ylim=None,log=False,linear=True):
    y = offdiag_elems(im1)
    x = offdiag_elems(im2)
    
    x_ = np.linspace(np.min(x),np.max(x),100)
    
    if log:
        coef,c,m,err_ = corr_coef(np.log(x),np.log(y),print_err=True)
        y_ = np.exp(c)*x_**m
    else:
        coef,c,m,err_ = corr_coef(x,y,print_err=True)
        y_ = c+x_*m
    print(coef,c,m,err_)
    
    
    if linear:
        y_ = x_
    fig1, ax1 = plt.subplots(facecolor='white',figsize=(7,7))
    if log:
        ax1.loglog(x,y,'o',color='gray')
        ax1.loglog(x_,y_,'r',lw=2)
    else:
        ax1.plot(x,y,'o',color='gray')
        ax1.plot(x_,y_,'r',lw=2)
    
    #plt.axis('equal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(xlim)
    if title is None:
        title = "Coef:%s , offset:%s ,slope:%s,error:%s \n"%(coef,c,m,err_)
    plt.title(title)
    return fig1
def set_diag(im,value=np.nan):
    im_=np.array(im)
    for i in range(len(im_)):
        im_[i,i]=value
    return im_
def load_HIC(fl = r'C:\Users\Bogdan\juicebox\IMR90Norm',min_chr = 29372319,max_chr = 30602321,res =30000):
    """Given file as downloaded from JuiceBox <fl> in a chromosome, <min_chr>,<max_chr> and resolution <res> this returns a matrix with HiC reads
    #chr21:29372319-30602321 is the main region of interest in hg19
    """
    
    min_chr_ = min_chr-res
    dim  = (max_chr-min_chr_)/res
    imHIC=np.zeros([dim,dim])
    coords = np.loadtxt(fl)
    for coord_i,coord_j,ct in coords:
        coord_i_norm = int((coord_i - min_chr_)/res)
        coord_j_norm = int((coord_j - min_chr_)/res)
        if (coord_i_norm>=0)&(coord_i_norm<dim):
            if (coord_j_norm>=0)&(coord_j_norm<dim):
                if coord_i_norm!=coord_j_norm:
                    imHIC[coord_i_norm,coord_j_norm]+=ct
                    imHIC[coord_j_norm,coord_i_norm]+=ct
    for i in range(len(imHIC)):
        imHIC[i,i]=np.max(imHIC)
    return imHIC[1:,1:]
def flatten(list_):
    return [item for sublist in list_ for item in sublist]
def set_diag(im,value=np.nan):
    im_=np.array(im)
    for i in range(len(im_)):
        im_[i,i]=value
    return im_
def nan_gaussian_filter(mat,sigma,keep_nan=True):
    from scipy.ndimage import gaussian_filter
    U=np.array(mat)
    Unan = np.isnan(U)
    V=U.copy()
    V[U!=U]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[U!=U]=0
    WW=gaussian_filter(W,sigma=sigma)

    Z=VV/WW
    if keep_nan:
        Z[Unan]=np.nan
    return Z
def interp1dnan(A):
    #nan = np.nan
    #A = np.array([1, nan, nan, 2, 2, nan, 3,nan])
    A_=np.array(A)
    ok = np.isnan(A)==False
    xp = ok.ravel().nonzero()[0]
    fp = A[ok]
    x  = np.isnan(A).ravel().nonzero()[0]

    A_[np.isnan(A)] = np.interp(x, xp, fp)
    return A_


## Plotting tools
def fig_no_axis(**kwargs):
    """Get figure with no right/upper axis and which allows for saving text in pdf and larger font."""
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.size']=22
    fig1, ax1 = plt.subplots(facecolor='white',**kwargs)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    return fig1

def to_maps(zxys):
    from scipy.spatial.distance import pdist,squareform
    mats=np.array(list(map(squareform,list(map(pdist,zxys)))))
    return mats
    
def nan_corr_coef(x_,y_):
    x=np.ravel(x_)
    y=np.ravel(y_)
    keep=(np.isinf(x)==False)&(np.isinf(x)==False)&(np.isnan(x)==False)&(np.isnan(y)==False)
    x=x[keep]
    y=y[keep]
    return np.corrcoef([x,y])[0,1]

def corr_coef(x_,y_,print_err=False):
    x=np.ravel(x_)
    y=np.ravel(y_)
    keep=(np.abs(x)!=np.inf)&(np.abs(y)!=np.inf)&(np.isnan(x)==False)&(np.isnan(y)==False)
    x=x[keep]
    y=y[keep]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    if print_err:
        model = sm.OLS(y,A)
        result = model.fit()
        return np.corrcoef([x,y])[0,1],c,m,result.bse
    return np.corrcoef([x,y])[0,1],c,m

## Compartment analysis tools

def pca_components(im_cor):
    """returns the evals, evecs sorted by relevance"""
    from scipy import linalg as la
    data = im_cor.copy()
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    #evecs_red = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return  evals, evecs#, np.dot(evecs_red.T, data.T).T
def get_AB_boundaries(im_cor,evec,sz_min = 1,plt_val=False):
    vec = np.dot(evec,im_cor)
    vec_ = np.array(vec)
    vec_[vec_==0]=10.**(-6)
    def get_bds_sign(vec_s):
        val_prev = vec_s[0]
        bds_ = []
        for pos,val in enumerate(vec_s):
            if val!=val_prev:
                bds_.append(pos)
                val_prev = val
        return np.array(bds_)

    vec_s = np.sign(vec_)
    bds_ = get_bds_sign(vec_s)
    vec_ss = vec_s.copy()
    bds_ext = np.concatenate([[0],bds_,[len(vec_s)]])
    for i in range(len(bds_ext)-1):
        if bds_ext[i+1]-bds_ext[i]<sz_min:
            vec_ss[bds_ext[i]:bds_ext[i+1]]=0    
    
    first_val = vec_ss[vec_ss!=0][0]
    vec_ss_ = []
    for vvec in vec_ss:
        if vvec==0:
            if len(vec_ss_)>0:
                vec_ss_.append(vec_ss_[-1])
            else:
                vec_ss_.append(first_val)
        else:
            vec_ss_.append(vvec)
    bds = get_bds_sign(vec_ss_)
    bds_score = []
    bds_ext = np.concatenate([[0],bds,[len(vec)]])
    for i in range(len(bds)):
        lpca = np.median(vec[bds_ext[i]:bds_ext[i+1]])
        rpca = np.median(vec[bds_ext[i+1]:bds_ext[i+2]])
        #print lpca,rpca
        bds_score.append(np.abs(lpca-rpca))
    if plt_val:
        plt.figure()
        plt.title('A/B pca 1 projection')
        plt.plot(vec,'ro-')
        plt.plot(bds,vec[bds],'go')
        plt.show()
    return bds,bds_score
def get_gen_pos_gen_vals(mats):
    """Given list of single cell distance matrices, find the population-average median, then group based on genomic distance and compute medians across groups.
    Perform fit in log space 
    """
    im_dist = np.nanmedian(mats,0)
    gen_pos,gen_vals = get_norm(im_dist,func=np.nanmedian)
    ro,c,m=corr_coef(np.log(gen_pos),np.log(gen_vals))
    gen_vals = np.exp(c)*gen_pos**m
    return gen_pos,gen_vals
def nan_corrcoef(x,y):
    x_ = np.array(x)
    y_ = np.array(y)
    keep = (np.isinf(x_)==False)&(np.isinf(y_)==False)&(np.isnan(x_)==False)&(np.isnan(y_)==False)
    if np.sum(keep)>2:
        return np.corrcoef(x_[keep],y_[keep])[0,1]
    return 0
def cor_mat(im_log):
    im_log = np.array(im_log)
    im_cor = np.zeros(im_log.shape)
    for i in range(len(im_cor)):
        for j in range(i+1):
            im_cor[i,j]=nan_corrcoef(im_log[i],im_log[j])
            im_cor[j,i]=im_cor[i,j]
    return im_cor
def get_cor_matrix(mat,gen_pos=None,gen_vals=None,plt_val=True):
    mat_ = np.array(mat)
    
    if plt_val:
        plt.figure()
        plt.title('original distance matrix')
        plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()
    mat_[list(range(len(mat_))),list(range(len(mat_)))]=np.nan
    
    #normalize for polymer effect
    if gen_pos is not None:
        mat_ = perform_norm(mat_,gen_pos,gen_vals)
        #mat_ = np.log(mat_)
    else:
        #mat_ = np.log(mat_)
        pass
    mat_[np.isinf(mat_)]=np.nan
    if plt_val:
        plt.figure()
        plt.title('distance normalized matrix')
        plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()  

    #compute correlation matrix
    mat_ = cor_mat(mat_)

    if plt_val:
        plt.figure()
        plt.title('correlation matrix')
        plt.imshow(mat_,interpolation='nearest',cmap='seismic')
        plt.colorbar()
        plt.show()
    return mat_

## Tad insulation tools defined as in Nagano et al., Nature, 2017
def get_b(i,j):
    """returns the in-tad coordinates"""
    i,j = np.sort([i,j])
    bx,by=[],[]
    for y_ in range(i,j+1):
        for x_ in range(i,y_):
            bx.append(x_)
            by.append(y_)
    return bx,by
def get_a(i,j,k):
    i,j,k = np.sort([i,j,k])
    ax,ay=[],[]
    for x_ in range(i,j):
        for y_ in range(j+1,k+1):
            ax.append(x_)
            ay.append(y_)
    return ax,ay
def insulation(i,j,k,mat,tag='distance',func = np.nanmedian):
    mat_ = np.array(mat,dtype=float)
    xmax,ymax = mat_.shape
    max_ = min(xmax,ymax)-1
    i,j,k = min(i,max_),min(j,max_),min(k,max_)
    b1x,b1y = get_b(i,j)
    b2x,b2y = get_b(j,k)
    ax,ay = get_a(i,j,k)
    B1 = func(mat_[b1x,b1y])
    B2 = func(mat_[b2x,b2y])
    A = func(mat_[ax,ay])
    if tag=='contact':
        return np.log(1+B1/A+B2/A)
    elif tag=='distance':
        return np.log(1+A/B1+A/B2)/np.log(2)


## Diffraction-limited/STORM single-cell boundary

def get_ind_loc_max(ratio,cutoff_max=1.2,valley=3):
    ratio_ext = np.array([1]+list(ratio)+[1])
    local_max = np.where(((ratio_ext[1:-1]-ratio_ext[:-2])>0)&((ratio_ext[1:-1]-ratio_ext[2:])>0))[0]
    
    local_max_good = local_max[ratio[local_max]>cutoff_max]
    local_max_good_=[]
    for id_ in local_max_good:
        l = np.max([0,id_-valley])
        r = np.min([len(ratio),id_+valley])
        if ratio[id_]==np.nanmax(ratio[l:r]):
            local_max_good_.append(id_)
    return np.array(local_max_good_)
def get_boundaries(im,su=4,sl=2,cutoff_max=1.2,valley=3,gb=0.75,func=np.nanmean,plt_val=False,no_pts=False):
    img = nan_gaussian_filter(im,gb,keep_nan=True)
    im_=img
    for i in range(len(im_)):
        im_[i,i]=np.nan
    ratio,ration,center,centern=[],[],[],[]
    ratio_o,ration_o=[],[]
    for i in range(len(im)):
        x_im_l,y_im_l = [],[]
        x_im_r,y_im_r = [],[]

        xn_im_l,yn_im_l = [],[]
        xn_im_r,yn_im_r = [],[]

        for j in range(sl):
            xn_im_l.extend(i+j-np.arange(su))
            yn_im_l.extend([i+j]*su)
        for j in range(sl):
            xn_im_r.extend(i+j+sl-np.arange(su))
            yn_im_r.extend([i+j+sl]*su)

        for j in range(sl):
            x_im_l.extend(i+j+np.arange(su))
            y_im_l.extend([i+j]*su)
        for j in range(sl):
            x_im_r.extend(i+j+sl+np.arange(su))
            y_im_r.extend([i+j+sl]*su)

        x_im_l,y_im_l,x_im_r,y_im_r = list(map(np.array,[x_im_l,y_im_l,x_im_r,y_im_r]))
        xn_im_l,yn_im_l,xn_im_r,yn_im_r = list(map(np.array,[xn_im_l,yn_im_l,xn_im_r,yn_im_r]))

        in_image = np.all(x_im_l>=0) and np.all(x_im_r>=0) and np.all(y_im_l>=0) and np.all(y_im_r>=0)
        in_image = in_image and np.all(x_im_l<len(im)) and np.all(x_im_r<len(im)) and np.all(y_im_l<len(im)) and np.all(y_im_r<len(im))

        in_imagen = np.all(xn_im_l>=0) and np.all(xn_im_r>=0) and np.all(yn_im_l>=0) and np.all(yn_im_r>=0)
        in_imagen = in_imagen and np.all(xn_im_l<len(im)) and np.all(xn_im_r<len(im)) and np.all(yn_im_l<len(im)) and np.all(yn_im_r<len(im))
        if in_image:
            val_l,val_r = func(im_[x_im_l,y_im_l]),func(im_[x_im_r,y_im_r])
            #ratio.append(val_r/val_l)
            ratio.append(val_l/val_r)
            val_l,val_r = func(im[x_im_l,y_im_l]),func(im[x_im_r,y_im_r])
            ratio_o.append(val_l/val_r)
            center.append(i+sl)
        if in_imagen:
            val_l,val_r = func(im_[xn_im_l,yn_im_l]),func(im_[xn_im_r,yn_im_r])
            #ration.append(val_l/val_r)
            ration.append(val_r/val_l)
            
            val_l,val_r = func(im[xn_im_l,yn_im_l]),func(im[xn_im_r,yn_im_r])
            ration_o.append(val_r/val_l)
            centern.append(i+sl-1)
        if False:#i==9:
            plt.figure()
            plt.plot(xn_im_l,yn_im_l,'mo')
            plt.plot(xn_im_r,yn_im_r,'go')
            plt.plot(x_im_l,y_im_l,'ro')
            plt.plot(x_im_r,y_im_r,'bo')
            plt.imshow(im,interpolation='nearest',cmap='seismic')
            plt.show()
            print(x_im_l,y_im_l,x_im_r,y_im_r)

    center,ratio=np.array(center),np.array(ratio)
    centern,ration=np.array(centern),np.array(ration)

    local_max_good = get_ind_loc_max(ratio,cutoff_max=cutoff_max,valley=valley)
    local_max_goodn = get_ind_loc_max(ration,cutoff_max=cutoff_max,valley=valley)
    if plt_val:
        #plt.close('all')
        plt.figure(figsize=(12,7))
        #print local_max_good,local_max_goodn
        plt.plot(center,np.log(ratio),'o-')
        if len(local_max_good)>0:
            plt.plot(center[local_max_good],np.log(ratio[local_max_good]),'o')
        plt.plot(centern,np.log(ration),'o-')
        if len(local_max_goodn)>0:
            plt.plot(centern[local_max_goodn],np.log(ration[local_max_goodn]),'o')
        plt.show()
        fig, ax = plt.subplots(figsize=(12,7))
        if not no_pts:
            if len(local_max_good)>0:
                ax.plot(center[local_max_good],center[local_max_good],'go')
            if len(local_max_goodn)>0:
                ax.plot(centern[local_max_goodn],centern[local_max_goodn],'m>')
        cax = ax.imshow(set_diag(img,np.nanmax(img)),interpolation='nearest',cmap='bwr')#,vmax=1000,vmin=0)
        #ticks_ = np.linspace(0,0.8,4)
        cbar = fig.colorbar(cax)#,ticks=ticks_)
        #cbar.ax.set_yticklabels(map(str,np.round(ticks_**(1./0.15),2)))
        plt.show()
    if len(local_max_good)>0 and len(local_max_goodn)>0:
        return center[local_max_good],centern[local_max_goodn],ratio,ration,center,centern,ratio_o,ration_o
    else:
        return [None]*8


def StandardBoundaryAnalysis(tags,dic_zxys,gb=1,valley=1,su=10,sl=6,
                             coverage_th=0.5,bad_hybe_cov_th=0.5,
                             cols=[[[0,0,0],[0.5,0.5,0.5]]],overwrite=False,boundary_file = None,plt_val=True):
    
    from scipy.spatial.distance import pdist,squareform
    
    dic = {}
    #load boundary_file if exists
    if boundary_file is not None:
        if os.path.exists(boundary_file):
            dic = pickle.load(open(boundary_file,'rb'))
    
    for itag in range(len(tags)):
        tag = tags[itag]
        zxys=dic_zxys[tag]
        zxysf=np.array(zxys)
        if (tag not in dic) or overwrite:
            col = cols[itag]
            zxys=dic_zxys[tag]
            zxysf=np.array(zxys)
            #threshold on coverage
            if coverage_th is not None:
                coverage = 1-np.sum(np.isnan(zxysf[:,:,0]),1)/float(zxysf.shape[1])
                zxysf = zxysf[coverage>coverage_th]
            #interpolate over bad hybes
            if bad_hybe_cov_th is not None:
                zxysa = np.array([np.array(list(map(interp1dnan,zxys_.T))).T for zxys_ in zxysf])
                coverage = np.sum(np.isnan(zxysf[:,:,0]),0)/float(len(zxysf))
                bad_hybes = np.where(coverage>0.5)[0]
                zxysb = np.array(zxysf)
                zxysb[:,bad_hybes]=zxysa[:,bad_hybes]
                zxysf = zxysb
            #compute matrices
            mats=np.array(list(map(squareform,list(map(pdist,zxysf)))))

            #initialize variables
            #There are two types of boundaries: start-of-a-block and end-of-a-block boundaries: a and b
            as_,bs_=[],[]#positions of the boundaries
            rsa,rsb=[],[]#strengths of the boundarries
            bds_cells,scorebds_cells=[],[]#positions per single cells and scores per single cells
            for imat_,mat_ in enumerate(mats[:]):
                #exclude trivial 0-diagonal
                mat_[np.arange(len(mat_)),np.arange(len(mat_))]=np.nan
                #for distance
                a,b,ra,rb,ca,cb,rao,rbo = get_boundaries(mat_,su=su,sl=sl,cutoff_max=1.,
                                                              valley=valley,gb=gb,func=np.nanmean,plt_val=False,no_pts=False)
                if a is not None:
                    a=np.array(a)
                    b=np.array(b)
                    a=a[(a<np.max(ca))&(a>np.min(ca))]
                    b=b[(b<np.max(cb))&(b>np.min(cb))]

                    rsa_=np.array(rao)[[list(ca).index(a_)for a_ in a]]
                    rsb_=np.array(rbo)[[list(cb).index(a_)for a_ in b]]

                    dica=dict(list(zip(a-0.5,rsa_)))
                    dicb=dict(list(zip(b+0.5,rsb_)))
                    unk = np.union1d(a-0.5,b+0.5)
                    score = [np.nanmean([dica.get(unk_,np.nan),dicb.get(unk_,np.nan)]) for unk_ in unk]
                    rsa.extend(rsa_)
                    rsb.extend(rsb_)
                    as_.extend(a)
                    bs_.extend(b)
                    bds_cells.append(unk)
                    scorebds_cells.append(score)
                    #break
                    if False:
                        plt.figure()
                        plt.plot(ca,rao,'r')
                        plt.plot(a,np.array(rao)[[list(ca).index(a_)for a_ in a]],'ro')
                        plt.plot(cb,rbo,'b')
                        plt.plot(b,np.array(rbo)[[list(cb).index(a_)for a_ in b]],'bo')


            print("Number of chromosomes:",imat_)
            x,y=np.unique(as_[::],return_counts=True)
            x=x-0.5
            y=y/float(imat_)
            dic_start = {x_:y_ for x_,y_ in zip(x,y)}
            x,y=np.unique(bs_[::],return_counts=True)
            x=x+0.5
            y=y/float(imat_)
            dic_end = {x_:y_ for x_,y_ in zip(x,y)}
            xs = np.sort(list(dic_start.keys())+list(dic_end.keys()))
            means = [np.nanmean([dic_start.get(x_,np.nan),dic_end.get(x_,np.nan)]) 
                     for x_ in xs]
            rsa_=rsa#np.log(rsa)
            rs1=np.array(rsa_)[(np.isnan(rsa_)==False)&(np.isinf(rsa_)==False)]
            rsa_=rsb
            rs2=np.array(rsa_)[(np.isnan(rsa_)==False)&(np.isinf(rsa_)==False)]
            rs = np.concatenate([rs1,rs2])


            dic[tag]={'xs':xs,'means':means,'rs':rs,'col':col,'bds_cells':bds_cells,'scorebds_cells':scorebds_cells}
        
        if plt_val:
            #### plots
            
            xs,means,rs,col = dic[tag]['xs'],dic[tag]['means'],dic[tag]['rs'],dic[tag]['col']
            fig_no_axis()
            plt.title(tag)
            plt.plot(xs,means,'o-',color=col[0])    
            plt.xlabel('Segment index')
            plt.ylabel('Probability')
            plt.xlim([0,len(zxysf[0])])
            plt.show()

            fig = fig_no_axis()
            plt.title(tag)
            vals,borders,_ =plt.hist(rs,bins=np.linspace(0.5,4,20),color=col[1])
            vls = np.linspace(0,np.max(vals),5)
            plt.yticks(vls,np.round(vls/len(rs),2))
            plt.xlabel('Boundary strength')
            plt.ylabel('Probability')
            plt.show()
        if boundary_file is not None:
            pickle.dump(dic,open(boundary_file,'wb'))
    return dic



## Diffraction-limited/STORM separation score
def separation_score(xyz1,xyz2,plt_val=False):
    #xyz1 = np.random.random([10,2])
    #xyz2= np.random.random([20,2])+0.5
    c1 = np.nanmedian(xyz1,0)
    c2 = np.nanmedian(xyz2,0)
    #c0 = np.nanmean(np.concatenate([xyz1,xyz2],0),0)
    #c0 = (c1*len(xyz1)+c2*len(xyz2))/(len(xyz1)+len(xyz2))
    c0=(c1+c2)/2.
    cdif = c2-c1
    bad1 =np.sum(np.dot(xyz1-c0,cdif)>0) 
    bad2 = np.sum(np.dot(xyz2-c0,cdif)<0)
    bad =(bad1 +bad2)/float(len(xyz1)+len(xyz2))

    if plt_val:
        x1,y1 = np.array(xyz1).T[:2]
        x2,y2 = np.array(xyz2).T[:2]
        plt.plot([c0[0]-(c1-c2)[1]*1,c0[0]+(c1-c2)[1]*1],[c0[1]+(c1-c2)[0]*1,c0[1]-(c1-c2)[0]*1],'k-')
        plt.plot(x1,y1,'ro')
        plt.plot(x2,y2,'bo')
        plt.title('Red bad:'+str(bad1)+', blue bad:'+str(bad2))
    return 1-bad

## Loops
def prob_loops(A1,A2,B1,B2,dist_mat,cut_off=150,nonan=True):
    """Given loopA nodes A1,A2 and loopB nodes B1,B2 a distance matrix <dist_mat> of length no_cells x nr_regions x nr_regions and a cutoff distance <cut_off>
    This returns p(A|B),p(A|notB),p(A),p(B)"""
    A_d = np.array(dist_mat[:,A1,A2])
    B_d = np.array(dist_mat[:,B1,B2])
    if nonan:
        keep = (np.isnan(A_d)|np.isnan(B_d))==False
        A_d=A_d[keep]
        B_d=B_d[keep]
    Ac=(A_d<cut_off)
    Bc=(B_d<cut_off)
    Bnc = Bc==False
    if len(Ac)>0:
        pAgB = np.sum(Ac&Bc)/float(np.sum(Bc))
        pAgnB = np.sum(Ac&Bnc)/float(np.sum(Bnc))
        pA,pB = float(np.sum(Ac))/len(Ac),float(np.sum(Bc))/len(Bc)
        pAB = np.sum(Ac&Bc)/float(len(Ac))
        return pAgB,pAgnB,pA,pB,pAB
    return [np.nan]*5
def get_probs(loops,dist_mat,cut_off=150,shift=0):
    """
    Given a set of <loops> and a set of single-cell distance matrices <dist_mat>
    consider contacts as distance<cut_off (typically 150nm) and return the probabilities/loop pair that 
    p(loop A|loop B),p(loop A|not loop B),p(A)
    """
    pAgBs=[]
    pAgnBs=[]
    pAs=[]
    pBs=[]
    nlim = len(dist_mat[0])
    for A1,A2,B1,B2 in np.array(loops)+shift:
        if A1<nlim and A2<nlim and B1<nlim and B2<nlim:
            pAgB,pAgnB,pA,pB,pAB = prob_loops(A1,A2,B1,B2,dist_mat,cut_off=cut_off)
            pAgBs.append(pAgB)
            pAgnBs.append(pAgnB)
            pAs.append(pA)
            pBs.append(pB)
    return [np.array(pAgBs),np.array(pAgnBs),np.array(pAs)]
def get_loopsall(nnodes,dmin = 0):
    nodes2 = list(range(nnodes))
    from itertools import combinations
    cmbs = list(combinations(nodes2,3))
    def cmb_to_loop_1(trp):
        return (trp[0],trp[1],trp[1],trp[2])
    def cmb_to_loop_2(trp):
        return (trp[2],trp[1],trp[1],trp[0])
    loopsall = list(map(cmb_to_loop_1,cmbs))+list(map(cmb_to_loop_2,cmbs))
    loopsall = list(map(cmb_to_loop_1,cmbs))+list(map(cmb_to_loop_2,cmbs))
    loopsall = [lp for lp in loopsall if np.abs(lp[0]-lp[1])>dmin and np.abs(lp[3]-lp[2])>dmin]
    return loopsall

## STORM specific functions
from scipy.spatial.distance import cdist
def overlap_metric(mlist_cluster1,mlist_cluster2,dist=200,error_tol=0.05,num_cutoff=1,kdtree=False,norm_tag='mean'):
    """Given two lists of mlists compute the overlap fraction"""
    xyz1 = np.array(cluster_to_xyz(mlist_cluster1,hlim=0,nmin=0,z_cut_off=500)).T
    xyz2 = np.array(cluster_to_xyz(mlist_cluster2,hlim=0,nmin=0,z_cut_off=500)).T
    if len(xyz1)==0 or len(xyz2)==0:
        return np.nan
    else:
        if not kdtree:
            dist_mat = cdist(xyz1,xyz2)
            th = dist_mat<dist
            fr1 = np.sum(th,axis=0)
            fr1 = np.sum(fr1>=num_cutoff)/float(len(fr1))
            fr2 = np.sum(th,axis=1)
            fr2 = np.sum(fr2>=num_cutoff)/float(len(fr2))
            return np.mean([fr1,fr2])
        else:
            tree = spatial.KDTree(xyz1)
            neighs = tree.query_ball_point(xyz2,dist,eps=error_tol)
            norm = float(len(xyz1)+len(xyz2))/2.
            frac_overlap = np.sum(np.array(list(map(len,neighs)))>=num_cutoff)/norm
            return frac_overlap            
def spotfiles_to_map(save_spots,metric_dic={'name':'dist'},reversed_=False,dim=3):
    maps=[]
    for save_spot in save_spots:
        map_=spotfile_to_map(save_spot,metric_dic=metric_dic,reversed_=reversed_,dim=dim)
        if map_ is not None:
            maps.append(map_)
    return maps
def spotfiles_to_center(save_spots,func=np.median,reversed_=False):
    maps=[]
    for save_spot in save_spots:
        map_=spotfile_to_center(save_spot,func=func,reversed_=reversed_)
        if map_ is not None:
            maps.append(map_)
    return np.array(maps)
def spotfile_to_center(save_spot,func=np.median,reversed_=False):
    dic = pickle.load(open(save_spot,'rb'))
    if dic.get('finished',False):
        visited = dic.get('visited',[])
        center_dic = {}
        cluster_dic = {}
        for e in visited:
            cluster_dic[e]=[]
            if e in dic:
                if type(dic[e]) is dict:
                    if 'mlist_clusters' in dic[e]:
                        if dic[e].get('mlist_clusters',None) is not None:
                            cluster_dic[e] = dic[e]['mlist_clusters']
            center_dic[e] = cluster_to_center(cluster_dic[e],func=func)
        reg_dic = non_redundant_regions(visited,reversed_=reversed_)
        reg_keys = np.sort(list(reg_dic.keys()))
        #print reg_dic
        return [center_dic[reg_dic[reg_keys[i]]] for i in range(len(reg_keys))]
    return None
    
def hybe_to_region(hybe):
    """Get the region from the folder naming convention"""
    return int(hybe.split('R')[-1].split('T')[0].split('A')[0])
def hybe_to_hindex(hybe):
    """Get the hybe index from the folder naming convention"""
    return int(hybe.split('H')[-1].split('A')[0].split('R')[0])
def xyz_to_rg(pts):
    pts_ =np.array(pts)
    return np.sqrt(np.mean(np.sum((pts_-[np.median(pts_,axis=0)])**2,axis=-1)))
def non_redundant_regions(hybes,reversed_=False):
    """Get a dictionary which contains the non-repeat regions for each region"""
    hybe_list = list(hybes)
    hindeces = [hybe_to_hindex(hybe)for hybe in hybe_list]
    hybe_list_sorted = list(np.array(hybe_list)[np.argsort(hindeces)])
    reg_dic = {}
    for hybe in hybe_list_sorted:
        region = hybe_to_region(hybe)
        if reversed_:
            reg_dic[region] = hybe
        else:
            if not region in list(reg_dic.keys()):
                reg_dic[region] = hybe
    return reg_dic
def repeat_regions(hybes):
    """Get a dictionary which contains the repeat regions"""
    hybe_list = np.array(list(hybes))
    hindeces = np.array([hybe_to_hindex(hybe)for hybe in hybe_list])
    hregs = np.array([hybe_to_region(hybe)for hybe in hybe_list])
    unk,cts = np.unique(hregs,return_counts=True)
    reps_regs = unk[cts>1]
    rep_dic = {rep:[hybe_list[hregs==rep],hindeces[hregs==rep]] for rep in reps_regs}
    return rep_dic
def cluster_to_xyz(mlist_cluster,hlim=0,nmin=0,z_cut_off=500,return_h=False):
    xs,ys,zs,hs= [],[],[],[]    
    for mlist_ in mlist_cluster:
        keep = (mlist_['h']>hlim)&(np.abs(mlist_['z'])<=z_cut_off)
        if np.sum(keep)>nmin:
            xs.extend(mlist_['xc'][keep])
            ys.extend(mlist_['yc'][keep])
            zs.extend(mlist_['zc'][keep])
            hs.extend(mlist_['h'][keep])
        #print len(xs)
    if return_h:
        return np.array(xs),np.array(ys),np.array(zs),np.array(hs)
    return np.array(xs),np.array(ys),np.array(zs)
def cluster_to_center(mlist_cluster,func=np.median,hlim=0,nmin=0,z_cut_off=500):
    mlist_center = np.array([np.nan,np.nan,np.nan])
    if len(mlist_cluster)>0:
        xs,ys,zs = [],[],[]
        for mlist_ in mlist_cluster:
            keep = (mlist_['h']>hlim)&(np.abs(mlist_['z'])<=z_cut_off)
            if np.sum(keep)>nmin:
                xs.extend(mlist_['xc'][keep])
                ys.extend(mlist_['yc'][keep])
                zs.extend(mlist_['zc'][keep])
        if len(xs)>0:
            mlist_center = np.array(list(map(func,[xs,ys,zs])))
    return mlist_center
    
def distance_metric(mlist_cluster1,mlist_cluster2):
    """Given two lists of mlists compute the distance between their center of masses"""
    if len(mlist_cluster1)==0 or len(mlist_cluster2)==0:
        return np.nan
    else:
        xs,ys,zs = [],[],[]
        for mlist_ in mlist_cluster1:
            xs.extend(mlist_['xc'])
            ys.extend(mlist_['yc'])
            zs.extend(mlist_['zc'])
        mlist_center1 = np.array(list(map(np.median,[xs,ys,zs])))
        
        xs,ys,zs = [],[],[]
        for mlist_ in mlist_cluster2:
            xs.extend(mlist_['xc'])
            ys.extend(mlist_['yc'])
            zs.extend(mlist_['zc'])
        mlist_center2 = np.array(list(map(np.median,[xs,ys,zs])))
        return np.sqrt(np.sum((mlist_center1-mlist_center2)**2))
def rg(zxys_):
    keep = np.isnan(zxys_[:,0])==False
    zxys__ = zxys_[keep]
    zxys__ = zxys__-np.nanmean(zxys__,0)
    return np.sqrt(np.mean(np.sum(zxys__**2,-1)))
def pts_to_cm(pts):
    cms = []
    for pt_set in pts:
        cms.append([])
        for pt_ in pt_set:
            cms[-1].append(np.median(pt_,0))
    return np.array(cms)
    
   