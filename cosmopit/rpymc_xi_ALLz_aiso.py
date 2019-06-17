from numpy import *
from pylab import *
import sys,subprocess
from scipy import optimize
import fitting,galtools,numMath,mypymcLib
import pk_Class,cosmology
from scipy import signal,interpolate
# correct the h dependence
# use: rpymc_xi_ALLz_aiso.py 'd2' 'JC' 'ngc' 'NoPrior2015_5baiso' 20000 '' 0 20.0 150.0

save     = True
doplot   = True
who_give = sys.argv[1]
who_Nest = sys.argv[2]
xgc      = sys.argv[3]   # 'ngc'                 #
#rfit = [float(sys.argv[4]),float(sys.argv[5])] # [40.,200.] #
kaiser=True
galaxy=True
change_gauge=False
nmock=1
addPrior=sys.argv[4] # 'Prior' 'NoPrior'
zi_in = 2 # define z bins
niter=int(sys.argv[5])
weights=sys.argv[6] # '' 'No' 'NoStar' 'NoSee' 'NoFKP' 'NoZCP'  
i_chain=sys.argv[7] # Number of chains for RB test
xxistart,xxistop=20,150 #what I use 20,150 #60,130                                                                                                                         
noPriorb_bool=False # True: gives a Prior on b | False: gives not a Prior on b

zedge,znames,dz,zmid,zbins= galtools.NameZ2(minz=0.43,maxz=0.7,nbins=5)

#if savefile in list_done:                                                                                                                                                               
print 'just read Now'
toto = load('../../DATA/'+'DM_RH_measure_CORALPHA'+weights+'_'+who_give+'_'+who_Nest+'_12_multi_xibias_minuit_eff_10.0_1.0-40.0_CGFalse_'+weights+'_r_90.0-600'+weights+'.npz')
totoXI = load('../../DATA/'+'bias_eff_10.0_ls_minuit_'+xgc+'_range_1.0-40.0'+weights+'_DATA.npz') # '_DATA'

if xgc=='ngc': XGCpar='n'
elif xgc=='sgc': XGCpar='s'

wok = where( (toto['rN']>xxistart)&(toto['rN']<xxistop) )
yyycut,covcut = [],[]
for zi in range(5) : 
    xxxcut,yyyyyy,covcovcov = numMath.cut_data(toto['rN'],totoXI['xi_data'][zi],totoXI['covmatObs'][zi],wok)
    yyycut.append(yyyyyy)
    covcut.append(covcovcov)

yyycut,covcut = array(yyycut),array(covcut)

x      = concatenate(( xxxcut                 ,array([zmid[zi_in]])                ))
y      = concatenate(( yyycut[zi_in]          ,array([toto['rh'+XGCpar][zi_in]])   ))
covmat = numMath.addRCVal(covcut[zi_in]*(1000.)/(1000.-1.),val=toto['drh'+XGCpar][zi_in]**2)

xz = concatenate((tile(xxxcut,len(zmid)).reshape(len(zmid),len(xxxcut)).T ,array([zmid]) ),axis=0).T
yz = concatenate((yyycut.T ,array([toto['rh'+XGCpar]]) ),axis=0).T 
covmatz=zeros(( len(zmid),len(covmat),len(covmat) ))
for zi in range(len(zmid)): covmatz[zi] = numMath.addRCVal(covcut[zi],val=toto['drh'+XGCpar][zi]**2)

params1  =  pk_Class.cosmo_PlanckAll(giveExtra=True)
pkTheory =  pk_Class.theory(params1,zmid[zi_in])

om_fid,ol_fid=0.3156,1.-0.3156
#                                           oc_f = 0.1198
h_f,ob_f,oc_f,ns_f,As_f,ok_f=0.6727,0.02225,0.3156*0.6727**2-0.02225,0.9645,3.094,0.0

paramsFid = {
'ln10^{10}A_s': As_f,
'n_s': ns_f,
'omega_b': ob_f,
'omega_cdm': oc_f,
'h': h_f,
'Omega_k': ok_f,
}

pkTheoryFid = pk_Class.theory(paramsFid,zmid[zi_in],halofit=False) #,ExtraPars=True)


hrd_fid = h_f*pkTheoryFid.get_rs()
def alpha_iso_new(z,om=om_fid,ol=ol_fid,hrd=hrd_fid):
    #z = concatenate((array([0]),z))
    DV_rd_TRUE = 1./hrd * (  z/cosmology.EE(z,omegam=om,omegax=ol)*(cosmology.properdistance(array([0.0,z]),omegam=om,omegax=ol)[0][1])**2.  )**(1./3.)
    DV_rd_FIDU = 1./hrd_fid *(  z/cosmology.EE(z,omegam=om_fid,omegax=ol_fid)*(cosmology.properdistance(array([0.0,z]),omegam=om_fid,omegax=ol_fid)[0][1])**2.  )**(1./3.)
    return (DV_rd_TRUE/DV_rd_FIDU)

def alpha_iso_new_omok(z,om=om_fid,ok=ok_f,hrd=hrd_fid):
    DV_rd_TRUE =  1./hrd     *(  z/cosmology.EE(z,omegam=om,omegax=1-om-ok)*(cosmology.properdistance(array([0.0,z]),omegam=om_fid,omegax=1-om-ok)[0][1])**2.  )**(1./3.)
    DV_rd_FIDU =  1./hrd_fid *(  z/cosmology.EE(z,omegam=om_fid,omegax=1-om_fid-ok_f)*(cosmology.properdistance(array([0.0,z]),omegam=om_fid,omegax=1-om_fid-ok_f)[0][1])**2.  )**(1./3.)
    return (DV_rd_TRUE/DV_rd_FIDU)

res_xi = zeros((5,len(xxxcut)))
pkTheory_z = []
for zi in range(len(zmid)):
    pkTheory_z.append( pk_Class.theory(params1,z=zmid[zi],halofit=False) )

"""
def model(x,pars):
    om  = pars[5]
    if addPrior=='CMBonly2015_5bomok_hrd' or addPrior=='AllPrior2015_5bomok_hrd' or 'NoPrior_5bomok_hrd':
        ok  = pars[6]
        #print 'ok model 1'
    elif addPrior=='CMBonly2015_5bomol_hrd' or addPrior=='AllPrior2015_5bomol_hrd' or 'NoPrior_5bomol_hrd' :
        ol  = pars[6]
    hrd = pars[7]
    res_xi = zeros((5,len(x)))
    for zi in range(5):
        if addPrior=='CMBonly2015_5bomok_hrd' or addPrior=='AllPrior2015_5bomok_hrd' or 'NoPrior_5bomok_hrd':
            res_xi[zi] = pars[zi]**2.*pkTheory_z[zi].xi(x/alpha_iso_new_omok(zmid[zi],om=om,ok=ok,hrd=hrd))
            #print 'ok model 2'
        elif addPrior=='CMBonly2015_5bomol_hrd' or addPrior=='AllPrior2015_5bomol_hrd' or 'NoPrior_5bomol_hrd':
            res_xi[zi] = pars[zi]**2.*pkTheory_z[zi].xi(x/alpha_iso_new(zmid[zi],om=om,ol=ol,hrd=hrd))
    res_xi[isnan(res_xi)] = -1e30
    return res_xi.flatten()
data = mypymcLib.Data(xvals=xxxcut.flatten(),yvals=yyycut.flatten(),errors=numMath.blockMat1row(covcut),model=model,nmock_prec=1000)
"""
def model_iso(x,pars):
    res_xi = zeros((5,len(x)))
    for zi in range(5):
        res_xi[zi] = pars[zi]**2.*pkTheory_z[zi].xi(x/pars[zi+5])
    res_xi[isnan(res_xi)] = -1e30
    return res_xi.flatten()

def model_5aiso2omolhrd(x,pars):
    return alpha_iso_new_omok(z,om=pars[0],ol=pars[1],hrd=pars[2])

def model_5aiso2omokhrd(x,pars):
    return alpha_iso_new_omok(z,om=pars[0],ok=pars[1],hrd=pars[2])


bb = 0.2 #0.1
nburn=0e3 #                                                                                                                                                              
if addPrior=='CMBonly2015_5b5aiso' or addPrior=='AllPrior2015_5b5aiso' or 'NoPrior_5b5aiso' or addPrior=='Prior_5b5aiso':
    variables = ['bias0','bias1','bias2','bias3','bias4','a0','a1','a2','a3','a4']
    data = mypymcLib.Data(xvals=xxxcut.flatten(),yvals=yyycut.flatten(),errors=numMath.blockMat1row(covcut),model=model_iso,nmock_prec=1000)
elif addPrior=='CMBonly2015_5bomok_hrd' or addPrior=='AllPrior2015_5bomok_hrd' or 'NoPrior_5bomok_hrd':
    variables = ['bias0', 'bias1', 'bias2', 'bias3', 'bias4','om','ok','hrd']
    data = mypymcLib.Data(xvals=xxxcut.flatten(),yvals=yyycut.flatten(),errors=numMath.blockMat1row(covcut),model=model,nmock_prec=1000)
elif addPrior=='CMBonly2015_5bomol_hrd' or addPrior=='AllPrior2015_5bomol_hrd' or 'NoPrior_5bomol_hrd':
    variables = ['bias0', 'bias1', 'bias2', 'bias3', 'bias4','om','ol','hrd']
    data = mypymcLib.Data(xvals=xxxcut.flatten(),yvals=yyycut.flatten(),errors=numMath.blockMat1row(covcut),model=model,nmock_prec=1000)
elif addPrior=='CMBonly2015_omol_hrd' or addPrior=='AllPrior2015_omol_hrd' or 'NoPrior_omol_hrd':
    variables = ['om','ol','hrd']
    data = mypymcLib.Data(xvals=zmid,yvals=DV_rd_MEASmean,errors=DV_rd_MEASstd,model=model,nmock_prec=1000)

print addPrior
print variables

if addPrior=='AllPrior2015k' or addPrior=='CMBonly2015k' or addPrior=='CMBonly2015_5bomol_hrd' or addPrior=='AllPrior2015_5bomol_hrd' or addPrior=='CMBonly2015_5bomok_hrd' or addPrior=='AllPrior2015_5bomok_hrd' or 'NoPrior_5bomok_hrd' or addPrior=='CMBonly2015_5b5aiso' or addPrior=='AllPrior2015_5b5aiso' or 'NoPrior_5b5aiso' or addPrior=='CMBonly2015_5aiso2omolhrd' or addPrior=='AllPrior2015_5aiso2omolhrd' or 'NoPrior_5aiso2omolhrd' or addPrior=='CMBonly2015_5aiso2omokhrd' or addPrior=='AllPrior2015_5aiso2omokhrd' or 'NoPrior_5aiso2omokhrd' or addPrior=='Prior_5b5aiso': 

    def gauss_prior_cmb_2015k(x, pars):
        sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_oc,sig_ok=0.02,0.02,0.02,0.02,0.03,0.0015,0.041
        if noPriorb_bool:
            delta_pars = array([(pars[0] - 1.88 -bb  ),(pars[1] - 1.85 -bb  ),(pars[2] - 1.94 -bb  ),(pars[3] - 2.00 -bb),(pars[4] - 2.15 -bb),(pars[5] - 0.1198),(pars[6] + 0.04 )])
            cov_5bocok = array([
            [sig_b0**2,0,0,0,0,0,0],
            [0,sig_b1**2,0,0,0,0,0],
            [0,0,sig_b2**2,0,0,0,0],
            [0,0,0,sig_b3**2,0,0,0],
            [0,0,0,0,sig_b4**2,0,0],
            #[0,0,0,0,0,sig_oc**2,0],                                                                                                                                                 
            #[0,0,0,0,0,0,sig_ok**2]                                                                                                                                                   
            [0,0,0,0,0,sig_oc**2,0.25*sig_oc*sig_ok],## add correlation 0.25                                                                                                           
            [0,0,0,0,0,0.25*sig_oc*sig_ok,sig_ok**2] ## add correlation                                                                                                                
            ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bocok)),delta_pars)
        else:
            delta_pars = array([(pars[5] - 0.1198),(pars[6] + 0.04 )])
            cov_5bocok = array([
            #[sig_oc**2,0],                                                                                                                                                            
            #[0,sig_ok**2]                                                                                                                                                            
            [sig_oc**2,0.25*sig_oc*sig_ok],## add correlation 0.25                                                                                                                     
            [0.25*sig_oc*sig_ok,sig_ok**2] ## add correlation                                                                                                                         
            ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bocok)),delta_pars)

        #chi2 = chi2_omega_c + chi2_Omega_k + chi2_bias                                                                                                                                 
        return chi2

    def gauss_prior_cmb_2015_5bomolhrd(x,pars):
        #sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_om,sig_ol,sig_hrd=0.02,0.02,0.02,0.02,0.03,0.0091,1./sqrt(1./0.021**1 + 1./0.0091**2),1./sqrt(0.0066**2.+1./(0.01*149.28)**2.)
        sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_om,sig_ol,sig_hrd=0.02,0.02,0.02,0.02,0.03,0.09,0.7,sqrt(0.0066**2.+(0.01*149.28)**2.)
        if noPriorb_bool:
            delta_pars = array([(pars[0] - 1.88 -bb  ),(pars[1] - 1.85 -bb  ),(pars[2] - 1.94 -bb  ),(pars[3] - 2.00 -bb),(pars[4] - 2.15 -bb),(pars[5] - 0.3156),(pars[6] - (1.-0.3156),(pars[7]-0.6727*149.28) )])
            cov_5bomol_hrd = array([
            [sig_b0**2,0,0,0,0,0,0,0],
            [0,sig_b1**2,0,0,0,0,0,0],
            [0,0,sig_b2**2,0,0,0,0,0],
            [0,0,0,sig_b3**2,0,0,0,0],
            [0,0,0,0,sig_b4**2,0,0,0],
            [0,0,0,0,0,sig_om**2,0,0],                                                                                                                                                             
            [0,0,0,0,0,0,sig_ol**2,0],
            [0,0,0,0,0,0,0,sig_hrd**2.]
            ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bomol_hrd)),delta_pars)
        else:
            delta_pars = array([(pars[5] - 0.3156),(pars[6] - (1-0.3156) ),(pars[7]-0.6727*149.28)])
            cov_5bomol_hrd = array([
                [sig_om**2,-0.95*sig_om*sig_ol,0],                                                                                                                                                 
                [-0.95*sig_om*sig_ol,sig_ol**2,0],
                [0,0,sig_hrd**2]
                 ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bomol_hrd)),delta_pars)
        return chi2

    def gauss_prior_cmb_2015_5bomokhrd(x,pars):
        sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_om,sig_ok,sig_hrd=0.02,0.02,0.02,0.02,0.03,0.09,0.02,sqrt(0.0066**2.+(0.01*149.28)**2.)
        if noPriorb_bool:
            delta_pars = array([(pars[0] - 1.88 -bb  ),(pars[1] - 1.85 -bb  ),(pars[2] - 1.94 -bb  ),(pars[3] - 2.00 -bb),(pars[4] - 2.15 -bb),(pars[5] - 0.3156),(pars[6] + 0.00),(pars[7]-0.6727*149.28 ) ])
            cov_5bomol_hrd = array([
            [sig_b0**2,0,0,0,0,0,0,0],
            [0,sig_b1**2,0,0,0,0,0,0],
            [0,0,sig_b2**2,0,0,0,0,0],
            [0,0,0,sig_b3**2,0,0,0,0],
            [0,0,0,0,sig_b4**2,0,0,0],
            [0,0,0,0,0,sig_om**2,0,0],
            [0,0,0,0,0,0,sig_ok**2,0],
            [0,0,0,0,0,0,0,sig_hrd**2.]
            ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bomol_hrd)),delta_pars)
        else:
            delta_pars = array([(pars[5] - 0.3156),(pars[6] + 0.0), (pars[7]-0.6727*149.28)])
            cov_5bomol_hrd = array([
                [sig_om**2,-0.98*sig_om*sig_ok,-0.95*sig_om*sig_hrd],
                [-0.98*sig_om*sig_ok,sig_ok**2,0.90*sig_ok*sig_hrd],
                [-0.95*sig_om*sig_hrd,0.90*sig_ok*sig_hrd,sig_hrd**2]
                 ])
            chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bomol_hrd)),delta_pars)
        return chi2

    def gauss_prior_cmb_2015_5aiso2omokhrd(x,pars):
        sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_om,sig_ok,sig_hrd=0.02,0.02,0.02,0.02,0.03,0.09,0.02,sqrt(0.0066**2.+(0.01*149.28)**2.)
        delta_pars = array([(pars[0] - 0.3156),(pars[1] + 0.0), (pars[2]-0.6727*149.28)])
        cov_5bomol_hrd = array([
                [sig_om**2,-0.98*sig_om*sig_ok,-0.95*sig_om*sig_hrd],
                [-0.98*sig_om*sig_ok,sig_ok**2,0.90*sig_ok*sig_hrd],
                [-0.95*sig_om*sig_hrd,0.90*sig_ok*sig_hrd,sig_hrd**2]
                 ])
        chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5bomol_hrd)),delta_pars)
        return chi2

    def gauss_prior_5b5aiso(x,pars):
        sig_b0,sig_b1,sig_b2,sig_b3,sig_b4,sig_a0,sig_a1,sig_a2,sig_a3,sig_a4=0.02,0.02,0.02,0.02,0.03,  0.2,0.2,0.2,0.2,0.2
        delta_pars = array([(pars[0] - 1.88 -bb  ),(pars[1] - 1.85 -bb  ),(pars[2] - 1.94 -bb  ),(pars[3] - 2.00 -bb),(pars[4] - 2.15 -bb),(pars[5] - 1.0),(pars[6] - 1.0),(pars[7] - 1.0 ),(pars[8] - 1.0 ),(pars[9] - 1.0 ) ])
        cov_5b5aiso = array([
        [sig_b0**2,0,0,0,0,0,0,0,0,0],
        [0,sig_b1**2,0,0,0,0,0,0,0,0],
        [0,0,sig_b2**2,0,0,0,0,0,0,0],
        [0,0,0,sig_b3**2,0,0,0,0,0,0],
        [0,0,0,0,sig_b4**2,0,0,0,0,0],
        [0,0,0,0,0,sig_a0**2,0,0,0,0],
        [0,0,0,0,0,0,sig_a1**2,0,0,0],
        [0,0,0,0,0,0,0,sig_a2**2,0,0],
        [0,0,0,0,0,0,0,0,sig_a3**2,0],
        [0,0,0,0,0,0,0,0,0,sig_a4**2],
        ])
        chi2 = dot(dot(delta_pars,np.linalg.inv(cov_5b5aiso)),delta_pars)
        return chi2

    prior_cmb_2015k         = mypymcLib.Data(model=gauss_prior_cmb_2015k, prior=True)
    prior_cmb_2015_5bomolhrd = mypymcLib.Data(model=gauss_prior_cmb_2015_5bomolhrd, prior=True)
    prior_cmb_2015_5bomokhrd = mypymcLib.Data(model=gauss_prior_cmb_2015_5bomokhrd, prior=True)
    prior_cmb_2015_5aiso2omokhrd = mypymcLib.Data(model=gauss_prior_cmb_2015_5aiso2omokhrd, prior=True)
    prior_5b5aiso = mypymcLib.Data(model=gauss_prior_5b5aiso, prior=True)

    if addPrior=='AllPrior2015k':
        chains    = mypymcLib.run_mcmc([prior_cmb_2015k,data], variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bocdmOk')
    elif addPrior=='CMBonly2015k':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015k]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bocdmOk') 
    elif addPrior=='AllPrior2015_5bomol_hrd':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015_5bomolhrd,data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomolhrd')
    elif addPrior=='CMBonly2015_5bomol_hrd':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015_5bomolhrd]          , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomolhrd')
    elif addPrior=='AllPrior2015_5bomok_hrd':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015_5bomokhrd,data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomokhrd')
    elif addPrior=='CMBonly2015_5bomok_hrd':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015_5bomokhrd]          , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomokhrd')    
    elif addPrior=='CMBonly2015_5aiso2omokhrd':
        chains   = mypymcLib.run_mcmc([prior_cmb_2015_5aiso2omokhrd]      , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomokhrd')
    elif addPrior=='Prior_5b5aiso':
        chains   = mypymcLib.run_mcmc([prior_5b5aiso,data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5b5aiso')

if addPrior=='NoPriork':
    chains = mypymcLib.run_mcmc(data, variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bocdmOk')
elif addPrior=='NoPrior_5bomol_hrd':
    chains   = mypymcLib.run_mcmc([data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomolhrd')
elif addPrior=='NoPrior_5bomok_hrd':
    chains   = mypymcLib.run_mcmc([data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5bomokhrd')
elif addPrior=='NoPrior_5b5aiso':
    chains   = mypymcLib.run_mcmc([data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model='LCDMsimple_5b5aiso')

if noPriorb_bool:
    noPriorb_name=''
else:
    noPriorb_name='_noPriorb'
    noPriorb_name='_noPriorb0.2'

savez('Aiso'+noPriorb_name+'_correct_pk_Class_rpymc_xi_ALL_PL2015_z'+str(zi_in)+'_'+str(addPrior)+'_'+who_give+'_'+str(shape(variables)[0])+'pars_'+str(niter)+'_'+str(nburn)+'_'+xgc+weights+'_xxi_'+str(xxistart)+'_'+str(xxistop)+'_xizvaryW'+str(i_chain)+'.npz',chains=chains,vars=variables,galaxy=galaxy,kaiser=kaiser,niter=niter,nburn=nburn)

