import numpy as np
from numpy import *
import pymc
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
import numMath
# Taken from J.C.Hamilton and remodified by P.Ntelis October 2016

###############################################################################
########################## Monte-Carlo Markov-Chains Functions ################
###############################################################################
### define data classes
class Data():
    def __init__(self, xvals=None, yvals=None, errors=None, model=None, prior=False, nmock_prec=None):
        self.prior = prior
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        if not self.prior:
            if np.size(np.shape(errors)) == 1:
                self.covar=np.zeros((np.size(errors),np.size(errors)))
                self.covar[np.arange(np.size(errors)),np.arange(np.size(errors))]=errors**2
            else:
                self.covar = errors
            if nmock_prec!=None: self.covar = self.covar* (nmock_prec-1.)/(nmock_prec-len(self.xvals)-2.)
            self.invcov = np.linalg.inv(self.covar)
    
    def __call__(self,*pars):
        if  not self.prior:
            val=self.model(self.xvals,pars[0])
            chi2=np.dot(np.dot(self.yvals-val,self.invcov),self.yvals-val)
        else:
            chi2 = self.model(self.xvals, pars[0])
        return(-0.5*chi2)



#### MODELS and Bounds #####
Sfid_params_baiso = {
               'bias':2.0,
                'a':1.0
                }
def Sll_model_baiso(datasets, variables = ['bias','a'], fidvalues = Sfid_params_baiso):
    if (isinstance(datasets, list) is False): datasets=[datasets]

    bias     = pymc.Uniform('bias', 1.5,2.5, value = Sfid_params_baiso['bias'], observed = 'bias' not in variables)
    a        = pymc.Uniform('a', 0.7,1.3 ,   value = Sfid_params_baiso['a'],  observed = 'a'    not in variables)


    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias=bias,a=a):
        ll=0.
        pars = np.array([bias,a])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_omol = {
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                }
def Sll_model_omol(datasets, variables = ['om','ol'], fidvalues = Sfid_params_omol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om        = pymc.Uniform('om', -1.0,2.0 ,  value = Sfid_params_omol['om'],   observed = 'om'     not in variables)
    ol        = pymc.Uniform('ol', -1.0,2.0,   value = Sfid_params_omol['ol'],   observed = 'ol'     not in variables) # 0.090,0.300                                                                
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol):
        ll=0.
        pars = np.array([om,ol]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


#### END: MODELS and Bounds #####

def run_mcmc(data,niter=80000, nburn=20000, nthin=1, variables=['Om', 'Ol', 'w'], external=None, w_ll_model='bias_aiso',delay=1000):
    if w_ll_model=='LCDMsimple_omol':
        feed_ll_model = Sll_model_omol
        feedPars      = Sfid_params_omol
    elif w_ll_model=='bias_aiso':
        feed_ll_model = Sll_model_baiso
        feedPars      = Sfid_params_baiso
    chain = pymc.MCMC(feed_ll_model(data, variables, fidvalues=feedPars))
    chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,delay=delay)
    chain.sample(iter=niter,burn=nburn,thin=nthin)
    ch ={}
    for v in variables: ch[v] = chain.trace(v)[:]
    return ch

#### PLOTTING

def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,alpha=0.7,labels=None,Blabel=None,plotCorrCoef=True,plotScatter=False,NsigLim=3,ChangeLabel=False,Bpercentile=False,kk=0,plotNumberContours='12'):
    '''
    kk=5 # kk==0 gives all parameters
           kk!=0 removes kk first parameters
    plotNumberContours= '1', '2', '12'
                        '1'  1sigma contour 
                        '2'  2sigma contours 
                        '12' 1sigma and 2sigma contours
    '''
    nplots=len(vars)
    if labels is None: labels = vars
    if doit is None: doit=np.repeat([True],nplots)
    mm=np.zeros(nplots)
    ss=np.zeros(nplots)
    for i in np.arange(nplots):
        if vars[i] in chain.keys():
            mm[i]=np.mean( chain[vars[i]] )
            ss[i]=np.std(  chain[vars[i]] )
    if limits is None:
        limits=[]
        for i in np.arange(nplots):
            limits.append([mm[i]-NsigLim*ss[i],mm[i]+NsigLim*ss[i]]) # 3
    num=0
    for i in np.arange(nplots-kk)+kk:
         for j in np.arange(nplots-kk)+kk:
            num+=1
            if (i == j):
                a=subplot(nplots-kk,nplots-kk,num)
                a.tick_params(labelsize=8)
                if i == nplots-1: xlabel(labels[j],size=20)
                var=vars[j]
                if vars[i]=='bias': xlim( [mm[i]-20*ss[i],mm[i]+20*ss[i]] )
                else:               xlim( limits[i] )
                
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100 
                    bla=np.histogram(chain[var],bins=nbins,normed=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/5/(xhist[1]-xhist[0]))
                    if Bpercentile:
                        mm = np.mean(chain[var])
                        p25= np.percentile(chain[var],100-68) - mm 
                        p75= np.percentile(chain[var],68) - mm
                        plot(xhist,yhist/max(yhist),color=col,label='%0.2f $\pm_{%0.2f}^{+%0.2f}$'%(mm, p25,p75 ))
                    else:
                        plot(xhist,yhist/max(yhist),color=col,label='%0.3f $\pm$ %0.3f'%(np.mean(chain[var]), np.std(chain[var])))
                    ylim([0.,3.0])
                    legend(loc=1,frameon=False,fontsize=12) #8 15

            if (i>j):
                a=subplot(nplots-kk,nplots-kk,num)
                a.tick_params(labelsize=12) #8 15

                var0=labels[j]
                var1=labels[i]
                if vars[j]=='bias': xlim( [mm[j]-20*ss[j],mm[j]+20*ss[j]] )
                else:               xlim(limits[j])
                
                ylim(limits[i])
                if i == nplots-1: xlabel(var0,size=20)
                if j == 0+kk: ylabel(var1,size=20)    
                if (vars[i] in chain.keys()) and (vars[j] in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    if plotNumberContours=='12':
                        levels = [0.9545,0.6827]
                    elif plotNumberContours=='1':
                        levels = [0.6827]
                    elif plotNumberContours=='2':
                        levels = [0.9545]
                    a0=cont(chain[vars[j]],chain[vars[i]],levels=levels,color=col,nsmooth=sm,alpha=alpha,plotCorrCoef=plotCorrCoef,plotScatter=plotScatter) 
                

    if Blabel:
        frame1=plt.subplot(nplots,nplots,nplots) #,facecolor='white')
        frame1.plot(1,1,col,label=Blabel)
        frame1.legend(loc=1,numpoints=1,frameon=False,prop={'size':20}) #25 15
        frame1.set_frame_on(False)
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        frame1.axes.get_xaxis().set_ticks([]) 
        frame1.axes.get_yaxis().set_ticks([])        

    #subplots_adjust(wspace=0.3,hspace=0.1)
    #subplots_adjust(wspace=0.15,hspace=0.1)
    subplots_adjust(wspace=0.,hspace=0.)
    return(a0)
    
def getcols(color):
    color=str(color)
    if (color == 'blue') or (color == 'b'):
        cols=['SkyBlue','MediumBlue']
    elif (color == 'red') or (color == 'r'):
        cols=['LightCoral','Red']
    elif (color == 'green') or color == 'g':
        cols=['LightGreen','Green']
    elif (color == 'pink') or (color == 'm'):
        cols=['LightPink','HotPink']
    elif (color == 'yellow') or (color == 'y'):
        cols=['Yellow','Gold']
    elif (color == 'black') or (color == 'k'):
        cols=['grey','black']
    elif color == 'orange':
        cols=['Coral','OrangeRed']
    elif color == 'purple':
        cols=['Violet','DarkViolet']
    elif color == 'brown':
        cols=['BurlyWood','SaddleBrown']
    return(cols)

def cont(x,y,xlim=None,ylim=None,levels=[0.9545,0.6827],alpha=0.7,color='blue',nbins=256,nsmooth=4,Fill=True,plotCorrCoef=True,plotScatter=False,**kwargs):

    levels.sort()
    levels.reverse()
    cols=getcols(color)
    dx=np.max(x)-np.min(x)
    dy=np.max(y)-np.min(y)
    if xlim is None: xlim=[np.min(x)-dx/3,np.max(x)+dx/3]
    if ylim is None: ylim=[np.min(y)-dy/3,np.max(y)+dy/3]
    range=[xlim,ylim]

    a,xmap,ymap=scipy.histogram2d(x,y,bins=256,range=range)
    a=np.transpose(a)
    xmap=xmap[:-1]
    ymap=ymap[:-1]
    dx=xmap[1]-xmap[0]
    dy=ymap[1]-ymap[0]
    z=scipy.ndimage.filters.gaussian_filter(a,nsmooth)
    z=z/np.sum(z)/dx/dy
    sz=np.sort(z.flatten())[::-1]
    cumsz=integrate.cumtrapz(sz)
    cumsz=cumsz/max(cumsz)
    f=interpolate.interp1d(cumsz,np.arange(np.size(cumsz)))
    indices=f(levels).astype('int')
    vals=sz[indices].tolist()
    vals.append(np.max(sz))
    vals.sort()
    
    if Fill:
        for i in np.arange(np.size(levels)):
            contourf(xmap, ymap, z, vals[i:i+2],colors=cols[i],alpha=alpha,**kwargs)
    else:
        contour(xmap, ymap, z, vals[0:1],colors=cols[1],**kwargs)
        contour(xmap, ymap, z, vals[1:2],colors=cols[1],**kwargs)
    a=Rectangle((np.max(xmap),np.max(ymap)),0.1,0.1,fc=cols[1])
    
    if plotScatter: scatter(x,y,color=color,marker=u'.')
    if plotCorrCoef:
        mmx,ssx = x.mean(),x.std()
        mmy,ssy = y.mean(),y.std()
        rcorrcoeff = np.corrcoef(x,y)[0,1]
        #if abs(rcorrcoeff)>0.65:
        label_cont='r=%0.2f'%(rcorrcoeff)
        xarr = array([mmx-ssx,mmx+ssx])
        yarr = array([mmy-ssy,mmy+ssy])
        plot(xarr,xarr*0.0+mmy,color,label=label_cont)
        plot(xarr*0.0+mmx,yarr,color) 
        legend(loc=2,frameon=False,numpoints=1,fontsize=12) #8 15 20


    return(a)

###############################################################################
###############################################################################