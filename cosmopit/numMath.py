import numpy as np
from numpy import *
from scipy import linalg,sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import fitting
# Hellpfull numerics by P.Ntelis June 2014


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def randomarrayindices_production(nblines=50,randsize=10):
    print 'start of producing rand indices'
    randindices = np.random.rand(nblines)
    xx = np.argsort(randindices)
    randarrayindices= np.sort(xx[:int(randsize)])
    print(randarrayindices)
    return randarrayindices

def loadRand(fin,randarrayindices,qsize=4,randsize=10):
    ''' 
        takes a string name of a file with data fin
        (qsize x nblines) = ( number of columns ) x ( number of lines )     
        returns an array with randomly  
        sampling this file, with randsize-size

    '''
    res = np.zeros([np.shape(randarrayindices)[0],qsize])
    #print 'start loading file'
    i=0
    with open(fin) as fd:
        #print 'files is open!'
        for n, line in enumerate(fd):
            if n == randarrayindices[i]:
                res[i] = np.fromstring(line.strip(),sep=' ')
                print i,n,res[i]
                i = i + 1
                if i==randsize:
                    break
    return res


def covmat(x,y):
    """ return the covariane matrix of parameter x and y """
    dim    = len(x)
    z      = np.array([x,y])
    covmat = np.zeros((len(z),len(z)))
    for i in range(len(z)):
        for j in range(len(z)):
            covmat[i,j] =  sum( (z[i]-np.mean(x) )*(z[j]-np.mean(y)) ) / (dim-1.)
    return covmat

def blockMat1row(matr):
    temp = []
    for i in range(len(matr)):
        temp.append(matr[i])
    return sparse.block_diag(temp).toarray()

def rebinning(x):
    x_new = 0.5*(x[:-1]+x[1:])
    return x_new

def rebin_counts(x):
    x_new=zeros(len(x)-1)
    for i in range(len(x_new)): x_new[i] = x[i] + x[i+1]
    return x_new

def repeated(f, n):
    """ 
        Use: numMath.repeated( function , number_times)( arg )
    """
    def rfun(p):
        return reduce(lambda x, _: f(x), xrange(n), p)
    return rfun

def rebin_x(x,n):
    return x[0:len(x):n]

def rebin_dd3(dd):
    return dd[0:len(dd):3] + dd[1:len(dd):3] + dd[2:len(dd):3]

def rebinning_all(r,xi0,dd,rr,dr,n=1):
    r_new  =rebin_x(r,n+1)
    xi0_new=rebin_x(xi0,n+1)
    dd_new =rebin_x(dd,n+1) # rebin_dd3(dd) #
    dr_new =rebin_x(dr,n+1) # rebin_dd3(dr) #
    rr_new =rebin_x(rr,n+1) # rebin_dd3(rr) #

    return r_new,xi0_new,dd_new,rr_new,dr_new

def testArrBurn(arr,doplot=False,fmt='.',color='r',figN=3):
    kkk=0
    temp_m = zeros(arr.size-kkk)
    temp_s = zeros(arr.size-kkk)
    totalm = mean(arr)
    for i in arange(temp_m.size):
        temp_m[i] = mean(arr[:i+1])
        temp_s[i] = std(arr[:i+1]) #/kkk
    if doplot:
        plt.figure(figN),plt.clf()
        plt.subplot(311)
        plt.plot(arr,'b.')
        plt.plot(temp_m,color+'-',label='mean')
        plt.plot(temp_m+temp_s,color+'--',label='std')
        plt.legend(numpoints=1,frameon=False,loc=4)
        plt.subplot(312)
        plt.plot(temp_m/totalm,color+'-',label='mean')
        plt.legend(numpoints=1,frameon=False,loc=4)
        plt.plot(arange(temp_m.size),arange(temp_m.size)*0.+1.,'k--')
        plt.subplot(313)
        plt.plot(temp_s/totalm,color+'--',label='std/totalm')
        plt.legend(numpoints=1,frameon=False,loc=4)
        print totalm

    return temp_m,temp_s

def testMCMC_autocorrelation(theta,N_lag=20000,N_lag_start=500):
    ''' returns: 
        \rho_lag = \frac{  \sum^{N-lag}_i (\theta_i - mean(theta) )*(\theta_{i+lag} - \bar(\theta))  }{sum^{N}_i (\theta_i - \bar{\theta})^2}
    '''
    lag=arange(0,N_lag,N_lag_start)
    rho=zeros(( len(lag) ))
    mean_theta=mean(theta)
    rho_lag_denominator=sum( ( theta-mean_theta )**2 )
    for lag_i in range(len(lag)):
        for i in range(len(theta)-1-lag[lag_i]):
            rho[lag_i] += ( theta[i] - mean_theta )*(theta[i+lag[lag_i]] -mean_theta ) 
    return rho/rho_lag_denominator


def testMCMC_autocorrelationAll(toto):
    """ for all parameters """
    vars = toto['vars']
    chains = toto['chains'].item()
    autocorrelations=[]
    for variable in vars:
        autocorrelations.append(testMCMC_autocorrelation( chains[variable] )[0] )
    autocorrelations=array(autocorrelations)
    return autocorrelations

def compute_Rubyn(totos,nlag=0):
    """
        Rubyn Test                                                                                                                                                                                                                      
        input :   totos = array([CMBBAO0,CMBBAO1])                                                                                                                                                                                      
        outout:   Rubyn_test         : All computed rubin statistic for each parameter                                                                                                                                                  
                  vars               : the name of the parameter correposnding to each test                                                                                                                                             
                  means_means_thetas : the mean of the means of the parameters                                                                                                                                                          
                  arange(nlag)       :                                                                                                                                                                                                  
        To see convergence check the condition                                                                                                                                                                                          
        Rubyn_test - 1 < 0.03 or 0.01                                                                                                                                                                                                   
                                                                                                                                                                                                                                        
        use: testR = compute_Rubyn(totos)                                                                                                                                                                                               
        Check convergence:                                                                                                                                                                                                              
        testR -1. < 0.03 #loose                                                                                                                                                                                                         
        testR -1. < 0.01 #tight  
    """

    vars=totos[0]['vars']
    chains=[]
    for variable in vars:
        for i in range(len(totos)): 
            chains.append( totos[i]['chains'].item()[variable] )
    chains = array(chains)
    thetas=[]
    for i in range(len(vars)):
        thetas.append( chains[len(totos)*i:len(totos)*(i+1),:][:,-nlag:] )
    print shape(chains[len(totos)*i:len(totos)*(i+1),:][:,-nlag:])
    thetas=array(thetas)

    dimParams=shape(thetas)[0]
    dimChains=shape(thetas)[1]

    mean_thetas = []
    sigma_thetas= []
    for i in range(dimParams):
        for j in range(dimChains):
            mean_thetas.append( mean(thetas[i][j] ) )
            sigma_thetas.append( std(thetas[i][j]*dimParams/(dimParams-1.) ) )
    mean_thetas=array(mean_thetas)
    sigma_thetas=array(sigma_thetas)


    W = []
    for j in range(dimParams):
        W.append(  sum(sigma_thetas[len(totos)*j:len(totos)*(j+1)]**2)/dimChains  )
    W = array(W)

    means_means_thetas = []
    for j in range(dimParams):
        means_means_thetas.append( sum(mean_thetas[len(totos)*j:len(totos)*(j+1)])/dimChains )
    means_means_thetas = array(means_means_thetas)
    Beta = []
    for j in range(dimParams):
        Beta.append( sum( (mean_thetas[len(totos)*j:len(totos)*(j+1)] -  means_means_thetas[j])**2 ) )
    Beta = array(Beta)*dimParams/(dimChains-1.)

    VarTheta = (1-1/dimParams)*W + Beta/dimParams

    Rubyn_test = sqrt(VarTheta/W)

    return Rubyn_test,vars,means_means_thetas,arange(nlag)

def Test_Rubyn(totos,nlag=0):
    """                                                                                                                                                                                                                                 
        convergence check condition                                                                                                                                                                                                     
        Rubyn_test - 1 < 0.03 or 0.01                                                                                                                                                                                                   
                                                                                                                                                                                                                                       
        use: testR = compute_Rubyn(totos)                                                                                                                                                                                               
        Check convergence:                                                                                                                                                                                                              
        testR -1. < 0.03 #loose                                                                                                                                                                                                         
        testR -1. < 0.01 #tight                                                                                                                                                                                                         
    """
    Rubyn_test,variables,means_means_thetas,it_lag = compute_Rubyn(totos,nlag=nlag)
    print 'The RG-test '
    print variables,'loose: RG-1. < 0.03'
    print Rubyn_test-1. < 0.03
    print variables,'tight: RG-1. < 0.01'
    print Rubyn_test-1. < 0.01
    print Rubyn_test-1.

def merge_chains(totos,nlag=0,KMIN=0):
    ''' Merge Chains '''
    print """ 
    when used with totoPlot reinitiallise it as 
    Merged_chains = numMath.merge_chains(array)
    totoPlot(Merged_chains)
    """
    vars=totos[0]['vars']
    new_toto={}
    for ki in totos[0].keys():          
        if ki=='data': 
            pass
        elif ki=='chains':
            new_toto.update( { str(ki): totos[0][str(ki)]  } )
            for parami in vars: new_toto[str(ki)].item()[parami] = zeros(( len(totos)*len(new_toto[str(ki)].item()[parami]) ))
        else: 
            new_toto.update( { str(ki): totos[0][str(ki)]  } )
        
    for param in totos[0]['chains'].item().keys():
        chain_list = []
        for i in range(len(totos)):
            chain_list.append(  totos[i]['chains'].item()[param][KMIN:] )
        new_toto['chains'].item()[param] = np.concatenate(chain_list)
    return new_toto

def burnChains(chains,kmin=0):
    ''' Try to find the bug '''
    newChains=dict(chains) # dict(chains)
    kmax = newChains[newChains.keys()[0]].size
    for k in newChains.keys(): newChains[k] = newChains[k][kmin:kmax]
    return newChains

def chainsTo2Darray(chain_in,PermMatlist=None):
    ''' 
        convert chain format to 2Darray 
        use: chain_out = numMath.chainsTo2Darray(chain_in,PermMatlist=None)
    '''
    chains = []
    for i in chain_in['chains'].item().keys(): chains.append(chain_in['chains'].item()[i])
    chains=array(chains)
    return chains[PermMatlist,:]

def average_realisations(datasim):
    ''' 
        general stat of a chain array: 
        out: means 
             stds
             covariance matrix
             correlation matrix
        use: means,stds,covMat,corMat=numMath.average_realisations(datasim)
        combine use:  (with the previous module: chainsTo2Darray)
chains_out = numMath.chainsTo2Darray(chain_in,PermMatlist=None)
means,stds,covMat,corMat=numMath.average_realisations(chains_out.T)
    '''
    dims=np.shape(datasim)
    nsim=dims[0]
    nbins=dims[1]
    meansim=np.zeros(nbins)
    sigsim=np.zeros(nbins)
    for i in np.arange(nbins):
        meansim[i]=np.mean(datasim[:,i])
        sigsim[i]=np.std(datasim[:,i])
    
    covmat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            covmat[i,j]=(1./(nsim))*np.sum((datasim[:,i]-meansim[i])*(datasim[:,j]-meansim[j]))

    cormat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            cormat[i,j]=covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])

    return(meansim,sigsim,covmat,cormat)

def insertP(arrmat):
    ''' inserts an additional column, and row with zeros on a matrix'''
    a_zero_c = np.zeros(len(arrmat))
    a_zero_r = np.zeros(len(arrmat)+1)
    arrmat_insert_c = np.insert(arrmat,len(arrmat),a_zero_c,axis=1)
    arrmat_insert_r = np.insert(arrmat_insert_c,len(arrmat_insert_c),a_zero_r,axis=0)
    return arrmat_insert_r

def addVal_Mat(arrmat,val=0.0):
    ''' adds a value on the last diagonal element of a matrix '''
    arrmat[len(arrmat)-1,len(arrmat)-1] =+ val
    return arrmat

def addRCVal(arrmat,val=0.0):
    ''' 
        add additional column and row and 
        puts a value on the last diagonal 
        of a symmtric matrix
    '''
    new_mat = insertP(arrmat)
    return addVal_Mat(new_mat,val=val)

def PermMat(mat,permut):
    ''' Permut and/or Removes Rows and Columns of a Matrix '''
    return (mat[permut,:])[:,permut]

def test_PermMat():
    testarr = array([['00','01','02','03','04'],['10','11','12','13','14'],['20','21','22','23','24'],['30','31','32','33','34'],['40','41','42','43','44']])
    print 'testarr'
    print testarr
    print 'remove column and row 1 1'
    print 'numMath.PermMat(testarr,[0,2,3,4])'
    print PermMat(testarr,[0,2,3,4])
    print 'permute column and row from 1 -> 3 and 3 -> 1'
    print 'numMath.PermMat(testarr,[0,3,2,1,4])'
    print PermMat(testarr,[0,3,2,1,4])

def AddColMat(mat,zeros=True):
    if zeros:
        newmat = np.zeros((mat.shape[0]+1,mat.shape[1]+1))
    else:
        newmat = np.ones((mat.shape[0]+1,mat.shape[1]+1))   
    newmat[:-1,:-1] = mat
    return newmat

def cut_data(r,Obs,covmat,wok):
    '''                                                                                                                                                                                  
        cuts data according to wok                                                                                                                                                       
        use: r_cut,Obs_cut,cov_cut = cut_data(r,Obs,covmat,wok)                                                                                                                          
    '''
    r_cut   = r[wok]
    Obs_cut = Obs[wok]
    cov_cut = (covmat[wok[0],:])[:,wok[0]]
    return r_cut,Obs_cut,cov_cut

def flattening_func(x,f,xlim=10,fval=None):
    """ flattens part of a function:
        f(x>xlim) = fval  
        or
        f(x>xlim) = mean(f(x>xlim))  """

    wok = np.where(x<xlim)
    wNok= np.where(x>xlim)

    if fval==None:
        fval = np.mean(f[wNok]) # take the average after xlim

    f_part= f[wNok]*0.0 + fval
    f_new = np.concatenate((f[wok],f_part)) 
    
    return f_new

def delta_kron(i,j):
    if i!=j: delta=0.0
    else: delta=1.0
    return delta
    
def open_cov(covmat,vector,kronecker=True):

    dims = np.shape(covmat)

    covmat_opened = covmat*0.0

    for i in np.arange(dims[0]):
        for j in np.arange(dims[1]):
            
            delta = 1.0
            if kronecker==True:
                delta=delta_kron(i,j)
            #print i,j,delta
            
            covmat_opened[i][j] = covmat[i][j] * (1. + vector[i]*vector[j]*delta )
    return covmat_opened

def vec2cov(vector):
    dims = np.shape(vector)
    covmat = np.zeros((dims[0],dims[0]))
    for i in np.arange(dims[0]):
        for j in np.arange(dims[0]):
            covmat[i][j]=vector[i]*vector[j]
    return covmat

def open_cov_Pierros(covmat,vector):
    ''' return covmat_opened = covmat * ( Ones + diag(covmat)*vector ) '''
    one_matrix = np.eye(np.shape(covmat)[0])
    correction = one_matrix*covmat*vector
    covmat_opened = covmat+correction
    return covmat_opened

def kth_diag_indices(matrix, k):
    '''how to take indices of matrix to select only kth-diagonals'''
    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

def cov_smooth(covmat):
    dim = np.shape(covmat)[0]

    mean_diag = np.zeros(dim)
    covmat_new = covmat*0.0 + covmat
    for i in xrange(dim):
        mean_diag[i]  = np.mean( np.diagonal(covmat,offset=i))

    for i in xrange(dim):
        if (i>0): # smooth all except the diagonal
            covmat_new[kth_diag_indices(covmat_new, i)] = mean_diag[None,i]
            covmat_new[kth_diag_indices(covmat_new, -i)] = mean_diag[None,i]

    return covmat_new

def give_msc(x,y,decimals=2):

    meanx,meany,stdx,stdy = np.around( np.mean(x), decimals=decimals) ,np.around( np.mean(y), decimals=decimals) ,np.around( np.std(x), decimals=decimals) ,np.around( np.std(y), decimals=decimals)

    covXY = np.around( covariance(x,y) , decimals=decimals )
    return meanx,meany,stdx,stdy,covXY

def plot_ms(x,y,xname='',yname='',unitsx='',unitsy='',addtitle='',zi=0,lenz=1,b=4.27,plotkk=False,doplot=True):

    if(plotkk):
        x2,x1 = np.max(x),np.min(x)
        y2,y1 = np.min(y),np.max(y)
        x_kk = np.linspace(x2,x1,100)
        a_kk = (y2-y1)/(x2-x1)
        y_kk = a_kk*x_kk + b
    
    meanx,meany,stdx,stdy,covXY = give_msc(x,y,decimals=2)

    #if (lenz==1): plt.suptitle(addtitle)
    if (doplot):
        plt.subplot(2,2*lenz,1+zi*2) #221
        plt.title(addtitle)
        plt.ylabel(yname+unitsy)
        plt.xlabel(xname+unitsx)
        if(plotkk): plt.plot(x_kk,y_kk,'k--')
        plt.scatter(x,y)
        
        plt.subplot(2,2*lenz,2+zi*2) #222
        plt.xlabel(yname+unitsy)
        plt.hist(y,color='b')
        plt.legend(loc=1, numpoints=1)
        
        plt.subplot(2,2*lenz,3+(zi+(lenz-1) )*2) #223
        plt.hist(x,color='g')
        plt.xlabel(xname+unitsx)
        plt.legend(loc=1, numpoints=1)

        plt.subplot(2,2*lenz,4 +(zi+(lenz-1) )*2) #224
        plt.plot(0,0,'b' ,label = yname+'='+str(meany)+'$\pm$'+str(stdy)+unitsy )
        plt.plot(0,0,'g' ,label = xname+'='+str(meanx)+'$\pm$'+str(stdx)+unitsx )
        plt.plot(0,0,'k' ,label = 'Cov['+xname+','+yname+'] = '+str(covXY) )
        plt.legend(loc='center', numpoints=1)
        plt.draw()

    return meanx,meany,stdx,stdy,covXY

def sigma_xy(x,y):
    ''' 
    returns sigma_xy , just covariance, no matrix 
    numpy.cov returns the whole covariance matrix
    numMath.sigma_xy(x,y) == np.cov(x,y)[0,1]
    '''
    meanx = np.mean(x)
    meany = np.mean(y)
    cov   = np.mean((x-meanx)*(y-meany))
    return cov

def corrc_xy(x,y):
    ''' 
    returns r_xy , just correlation coefficient, no matrix 
    numpy.corrcoeff returns the whole correlation coefficient matrix
    numMath.corrc_xy(x,y) == np.corrcoef(x,y)[0,1]
    '''
    num_sigma_xy = sigma_xy(x,y)
    num_corrc_xy = num_sigma_xy/(np.std(x)*np.std(y))
    return num_corrc_xy

def corrmat(covmat):
    dims = np.shape(covmat)
    cc = covmat*0.0

    for i in range(dims[0]):
        for j in range(dims[1]):
            cc[i][j] = covmat[i][j]/np.sqrt(covmat[i][i]*covmat[j][j])
    return cc

def stat2var(x,y):
    ''' return mx,sx,my,sy,cov(x,y),corr(x,y) '''
    return np.mean(x),np.std(x),np.mean(y),np.std(y),np.cov(x,y)[0,1],np.corrcoef(x,y)[0,1]

def covAverage(x,cov):
    """
    returns weight mean accoring to covariance matrix
    Gives the mean and std 
    accounting 
    for covariance matrix
    """
    Nx = len(x)
    W = np.ones(Nx)    
    invcov = linalg.inv(cov)
    var_x = 1./np.dot(W.T, np.dot(invcov,W))
    mean_x = var_x*np.dot(W.T, np.dot(invcov,x))
    std_x = np.sqrt(var_x)
    return(mean_x, std_x)

def plot3D_pierros(x,y,f_xy,zname='z=?',savename='plot3D_pierros.png',save=False):

    #X,Y = meshgrid(rS,rS)
    X,Y = x,y
    Z = f_xy
    print X , Y
    print Z
    print X.shape , Y.shape , Z.shape
    fig, ax = plt.subplots()
    plt.xlabel('$bias$',size=20)
    plt.ylabel('$\sigma_p\ [km\ s^{-1}]$',size=20)
    plt.suptitle(zname)
    #plt.zlabel('$Ratio(bias,\sigma_p) = R^{Distorted}_H(b,\sigma_p) / R_H$')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim(1.99,2.5)
    #plt.xlim(55,67)
    #plt.plot(x,y)
    p = ax.pcolor(X, Y, Z, cmap=cm.jet, vmin=np.min(Z), vmax=np.max(Z) )#, label='$R^{(r)}_H/R^{(s)}_H$')
    #p = ax.pcolor(X, Y, Z, cmap=cm.jet, vmin=np.min(Z), vmax=np.max(Z),label='$\frac{R^{(r)}_H}{R^{(s)}_H}$')
    cb = fig.colorbar(p, ax=ax,label='$\mathcal{R}^{(RSD)}_H/\mathcal{R}^{(linear)}_H$ in $\%$')

    if save == True:
        print 'Saving ...'
        plt.savefig(savename+'.png',dpi=100)

def plotScatter(x,y,f_xy,xxlabel='',yylabel='',zzlabel='',**kwargs):
    cm = plt.cm.get_cmap('rainbow')
    sc = plt.scatter(x,y, c=f_xy, s=35, cmap=cm)
    plt.ylabel(yylabel,size=25),plt.xlabel(xxlabel,size=25)
    plt.colorbar(sc).set_label(zzlabel,size=25)

def loadRand_old(fin,qsize=4,rbins=50,randsize=10):
    ''' 
    takes a string name of a file with data
    (col x rows) = (qsize x rbins) 
    returns an array(qsize x randsize) 
    with randomly sampling this file
    '''
    res = np.zeros([a.size,qsize])
    bsize = 0
    while bsize != randsize:
        a = np.random.random_integers(low=rbins, size=randsize)
        b = np.unique(a)
        bsize = b.size

    i=0
    with open(fin) as fd:
        for n, line in enumerate(fd):
            if (n in a):
                res[i] = np.fromstring(line.strip(),sep=' ')
                i = i + 1
    return res
    

def mySVD(matrix,doCheck=0,kk=False):
    '''
    Singular Value decomposition                                   
    method to compute the inverse
    of matrix        
    option: doCheck=0,1,2
    0: return only the inverted SVD matrix
    1: return previous plus composite matrices
    2: return previous plus checks
           0,1                         ,2
    USAGE: s,sU,sUt,sV,sVh,sSig,sinvSig,sCheck = numMath.mySVD(a,doCheck=2)
    '''
    ### Compute the SVD parts
    M,N       = matrix.shape
    U,s,Vh    = linalg.svd(matrix)
    Sig       = linalg.diagsvd(s,M,N)

    V = np.matrix(Vh).H
    Ut = np.matrix(U).T
    
    # invSig = linalg.inv( np.matrix(Sig) )
    invSig = np.matrix(Sig).I
    
    ### Correct for ill-ness
    w = np.where(Sig<=10**(-14))
    invSig[w]=0.0
    
    ### Compute the Inverse of the matrix 
    invSVD = V.dot(invSig).dot(Ut)  

    ### and check the matrix
    checkMatrix = U.dot(Sig.dot(Vh))
    checkProduct = invSVD.dot(matrix)

    #print ' invSVD = \n', invSVD
    #print ' sCheck = \n', checkMatrix

    #print ' invSVD.dot(matrix)     = \n  ', checkProduct

    if(doCheck==0):
        return(invSVD)
    elif(doCheck==1):
        return(invSVD,U,Ut,V,Vh,Sig,invSig)
    elif(doCheck==2):
        if(kk==True):  
            print 'V.dot(Vh)= \n',V.dot(Vh)
            print 'U.dot(Ut)= \n',U.dot(Ut)  
            print 'Sig.dot(invSig)= \n',Sig.dot(invSig)
        return(invSVD,U,Ut,V,Vh,Sig,invSig,checkMatrix)

    else:
        print'Read the description of numMath.mySVD'

################### Chi2 Robustness TEST for mocks ########################################### 
def Pierros_histogram(data_array,Normalization=True,numberBins=100):
    histo = np.histogram(data_array,bins=numberBins)
    x_center_hist = (histo[1][:-1]+histo[1][1:])/2
    y_hist = histo[0]
    yerror_hist = np.sqrt(y_hist)

    if Normalization==True:
        dx = histo[1][1] - histo[1][0]
        I_norm = np.sum(y_hist*dx)
        y_hist_normed = y_hist/I_norm
        error_hist_normed = yerror_hist/I_norm
        y_hist = y_hist_normed
        yerror_hist = error_hist_normed 
    return(x_center_hist,y_hist,yerror_hist)

def theoryChi2(x,pars):
    return( (1/(2**(pars[0]/2.)  * np.math.gamma(pars[0]/2.)) ) * x**(pars[0]/2. - 1) * np.exp( -x/2. ) )


def chi2_bias_test(chi2_mock,ndf=15,nbHistBins=4,method='minuit'):
    wok = np.where((chi2_mock<1600)&(chi2_mock>0))
    print np.max(chi2_mock)
    data = chi2_mock[wok]

    x_hist,y_hist,yerr_hist = Pierros_histogram(data,numberBins=nbHistBins) # 10

    guess = [np.float64(ndf)]
    res = fitting.dothefit(x_hist,y_hist,yerr_hist,guess,functname=theoryChi2,method=method)

    decimals = 2
    NDF_measured = np.around(res[1][0],decimals=decimals)
    dNDF_measured = np.around(res[2][0],decimals=decimals)
    chi2_measured = np.around(res[4],decimals=decimals)
    ndf_measured = np.around(res[5],decimals=decimals)

    # For plotting the theoretical chi2
    xmin,xmax = np.min(data),np.max(data) 
    xxx = np.linspace(xmin,xmax,1000)

    label_hist = "$ndf_{m} =$ "+str(NDF_measured)+"$\pm$"+str(dNDF_measured)+" at $\chi^2=$"+str(chi2_measured)+"/"+str(ndf_measured)
    plt.errorbar(x_hist,y_hist,yerr=yerr_hist,fmt='o',color='b',label=label_hist)
    plt.plot(xxx,theoryChi2(xxx,np.array([ndf])),'r-',label='ndf = '+str(ndf))
    plt.legend(loc=1)
################### END: Chi2 Robustness TEST for mocks ########################################### 

def pullCov(y,yth,cov):
    delta = y-yth
    eigenvals,rot = np.linalg.eigh(cov)
    newdelta = np.dot(rot.T,delta)
    pull = newdelta/np.sqrt( eigenvals )
    return pull

def pullDistr(x,x_err,x_mean=None,axis=None):
    """
    Tested on all zbins on Mocks North
    #1 pullN = ( x-np.average(x,weights=1./x_std**2) )/np.std(x, dtype=np.float64)
    #2 pullN = ( x-np.average(x,weights=1./x_std**2) )/x_std
    #3 pullN = ( x-np.mean(x) )/x_std
    1) -10e-13
    2) -10e-12
    3) -10e-13
    but the one ine use: -10e-14
    """
    if x_mean==None:
        if axis!=None:
            x_mean = np.mean(x,axis=axis)[:, None]
        else:
            x_mean = np.mean(x)
        pull = ( x-x_mean )/ ( x_err )
    else:
        pull = ( x-x_mean )/ ( x_err )
    return pull

def weight_mean(x_i,sig_i):

    weight_i = 1./sig_i**2.
    weight = sum(weight_i)
    weight_Norm_i = weight_i/weight

    mean_weight = sum( weight_i*x_i )

    return mean_weight   

def weight_std(x_i,sig_i):

    weight_i = 1./sig_i**2.
    weight = sum(weight_i)
    weight_Norm_i = weight_i/weight
    mean_weight = sum( weight_i*x_i )

    res = sum( weight_Norm_i*(x_i-mean_weight)**2. )
    
    return np.sqrt(res)

def NameZ2(minz,maxz,nbins):
    """
    returns 2 arrays which contain
    the edges of bins of redshift
    in double and string array 
    + 3 more arrays
    USE: zedge,znames,dz,zmid,nbins = numMath.NameZ2(minz=0.43,maxz=0.7,nbins=5)
    """
    zedge = np.linspace(minz,maxz,nbins+1)
    znames = np.array(np.zeros(nbins), dtype='|S20')
    
    for i in np.arange(nbins):
        znames[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])
    
    dz=np.zeros(nbins)+(zedge[1]-zedge[0])/2
    zmid=(zedge[np.arange(nbins)]+zedge[np.arange(nbins)+1])/2

    return(zedge,znames,dz,zmid,nbins)

def rebinMat(matrix):
    M,N = np.shape(matrix)
    return res

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def heaviside(x,x0=0,inverse=False):
    if inverse:
        delta = x0-x
    else:
        delta = x-x0
    res = 0.5*(np.sign(delta)+1)
    return res

def logBinning(r_max=1500.,nb_r=50,n_logint=10,a=1.):
    #a=1. #0.5                                                                                
    log_r_max=np.log10(r_max)
    powerL = np.zeros(nb_r)
    for i in xrange(nb_r):
            #radius[i] = 10**( ((i+a)-nb_r)/n_logint+log_r_max  )                             
            powerL[i] = ((i+a)-nb_r)/n_logint
            #radius[i]=(i+0.5)/(nb_r*(1./r_max)) # for regular binning                        
    radiusL=10**powerL * r_max
    return radiusL


def stat_realisations(datasim1,datasim2):
    """
    computes cross-covmat cross-corrmat
    for 2 variables 1 and 2
    coming from a simulation
    """
    dims=np.shape(datasim1)
    nsim=dims[1]
    nbins=dims[0]
    meansim1=np.zeros(nbins)
    sigsim1=np.zeros(nbins)
    meansim2=np.zeros(nbins)
    sigsim2=np.zeros(nbins)
    for i in np.arange(nbins):
        meansim1[i]=np.mean(datasim1[i,:])
        sigsim1[i]=np.std(datasim1[i,:])
    for i in np.arange(nbins):
        meansim2[i]=np.mean(datasim2[i,:])
        sigsim2[i]=np.std(datasim2[i,:])

    print '   stat:do covmat'
    covmat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            covmat[i,j]=np.mean((datasim1[i,:]-meansim1[i])*(datasim2[j,:]-meansim2[j]))

    print '   stat:do cormat'
    cormat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            cormat[i,j]=covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])

    return(meansim1,sigsim1,meansim2,sigsim2,covmat,cormat)

def plotCorrMat(x,y,datasim1,datasim2,savename='corrplot',save=False):
    print '  do stat'
    meansim1,sigsim1,meansim2,sigsim2,covmat,cormat = stat_realisations(datasim1,datasim2)
    
    #X,Y = meshgrid(rS,rS)
    X,Y = np.sort(x),np.sort(y)
    Z = cormat
    print X , Y
    print Z
    print X.shape , Y.shape , Z.shape
    fig, ax = plt.subplots()
    plt.ylabel('$bias$')
    plt.xlabel('$\mathcal{R}_H$ ($h^{-1}\ Mpc$)')
    ## plt.yscale('log')
    ## plt.xscale('log')
    #plt.ylim(1.99,2.5)
    #plt.xlim(55,67)
    plt.plot(x,y)
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    if save == True:
        print 'Saving ...'
        plt.savefig(savename+'.png',dpi=100)


# rounding indices inside a dictionary
class LessPrecise(float):
    def __repr__(self):
        return str(self)

def roundDict(d,num=4):
    for k, v in d.items():
        if isinstance( v, np.str ): pass
        else:
            v = LessPrecise(np.round(v, num))
            d[k] = v
    return d 
# rounding indices inside a dictionary
