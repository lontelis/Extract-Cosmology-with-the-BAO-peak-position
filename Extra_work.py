
### Computation of the significance of the detection of the BAO peak Position ###
### delta chi2 = chi2_Gaussian_PowerLaw - chi2_PowerLaw_Error
### Fit the Gaussian + PowerLaw
### Look Pierre Laurent Thesis, page 112, https://www.theses.fr/2016SACLS227.pdf
### Need to adjust the parameters to get a good measurement of the BAO Peak Position
### Take the significance level from Table 1 https://ned.ipac.caltech.edu/level5/Wall2/Wal3_4.html

def Gaussian(x,pars):
	return pars[1]*exp(-0.5*(x-pars[0]*r_dth)**2./pars[2]**2.)
def PowerLaw(x,pars):
	return pars[0] + pars[1]/x + pars[2]/x**2.
def Gaussian_PowerLaw(x,pars):
	return Gaussian(x, [pars[0],pars[1],pars[2]] ) + PowerLaw(x, [pars[3],pars[4],pars[5]] )

simulated_error_factor = 1.
guess = [1.0,0.005,10.,0.01,-2.,130.] # best guess, I found them by playing a bit on the Gaussian_PowerLaw, with simulated_error_factor = 1.
res_Gaussian_PowerLaw = fitting.dothefit(xx,yy,ye/simulated_error_factor,guess,functname=Gaussian_PowerLaw)#,parbounds=[(0.8,1.2),(.0,0.1),(0.0,0.3),(0.0,30.0),(-300.0,300.0),(0.0,100.0)])
res_PowerLaw_Error    = fitting.dothefit(xx,yy,ye/simulated_error_factor,guess,functname=PowerLaw_Error)#   ,parbounds=[(0.8,1.2),(-300.0,300.0),(-300.0,300.0),(-300.0,300.0),(-300.0,300.0),(-300.0,300.0)])

figure(3)
clf(),
errorbar(xx,yy*xx**2,ye/simulated_error_factor*xx**2,fmt='b.',label='DR12 CMASS sample')
plot(xx, Gaussian_PowerLaw(xx,res_Gaussian_PowerLaw[1])*xx**2.,'r-',label='Gaussian + PowerLaw, $a$=%0.3f$\pm$%0.3f'%(res_Gaussian_PowerLaw[1][0],res_Gaussian_PowerLaw[2][0]) )

plot(xx, PowerLaw_Error(xx,res_PowerLaw_Error[1])*xx**2.,'g-'      ,label='PowerLaw + Error   , $a$=%0.3f$\pm$%0.3f'%(res_PowerLaw_Error[1][0],res_PowerLaw_Error[2][0]) )

title('Factor of Simulated Error = %0.1f  \n $\Delta\chi^2 = \chi^2_{P} - \chi^2_{G+P} = $ %0.1f-%0.1f = %0.1f'%(simulated_error_factor,res_PowerLaw_Error[4],res_Gaussian_PowerLaw[4],res_PowerLaw_Error[4]-res_Gaussian_PowerLaw[4]))
legend()
ylabel('$s^2 \\xi_0(s) $ $[h^{-2}\mathrm{Mpc}^2]$')
xlabel('$s$ $[h^{-1} \mathrm{Mpc}]$')
draw(),
show()

###### Exercises for the student:

### Exercise 1: (Done)
### Fix the parameters of the power law to their best value, 
### Then redo the fit for the 3 rest parameters. 
### Compute the significance, give interpretation.

### Exercise 2: (Done)
### Fix the parameters of the power law and the Amplitude of the Gaussian to their best value, 
### Then redo the fit for the 2 rest parameters. 
### Compute the significance, give interpretation.

### Exercise 3: (Done)
### Fix the parameters of the power law and the dispersion of the Gaussian to their best value, 
### Then redo the fit for the 2 rest parameters. 
### Compute the significance, give interpretation.

### Exercise 4: 
### Fix the parameters of the power law and the Amplitude and dispertion of the Gaussian to their best value, 
### Then redo the fit for the 1 rest parameter. 
### Compute the significance, give interpretation.

### Exercise 5: 
###	Do the same exercises, using the covariance matrix now.

### Exercise 6:
### Do the same exercises for a a higher simulated error factor. What do you observe on the significance? 

###### New exercises for the student:

### Exercise 7: 
### Redo the modeling of PowerLaw + Error by only have the 3 PowerLaw parameters
### Construct the detection sigificance test as follows:
### Delta chi2( N-M number of parameters ) = chi2( Gaussian + PowerLaw, N=6 ) - chi2( PowerLaw, M=3 )
### Take the significance level from Table 1 https://ned.ipac.caltech.edu/level5/Wall2/Wal3_4.html

### Exercise 8:
### Do the previous exercise for a higher simulated error factor. What do you observe on the significance? 

####### Add CMB prior information

# Replace this line 
chains   = mypymcLib.run_mcmc(data, variables=variables,niter=niter,nburn=nburn,w_ll_model=' .... put your model name .... ')

#with :
def gauss_prior_cmb_2015_omolb(x,pars):
    om_cmb   = 0.48 
    ol_cmb   = 0.55
    rho_omol = -0.84
    sig_om,sig_ol=0.23/2.,0.17/2.
    delta_pars = array([(pars[0] - om_cmb),(pars[1] - ol_cmb )])
    cov_omolb = array([
    #[sig_oc**2,0],                                                                                                                                                            
    #[0,sig_ok**2]                                                                                                                                                            
    [sig_om**2,rho_omol*sig_om*sig_ol],## add correlation 0.25                                                                                                                     
    [rho_omol*sig_om*sig_ol,sig_ol**2] ## add correlation                                                                                                                         
    ])
    chi2 = dot(dot(delta_pars,np.linalg.inv(cov_omolb)),delta_pars)

    return chi2

prior_cmb_2015_omolb = mypymcLib.Data(model=gauss_prior_cmb_2015_omolb , prior=True)
chains   = mypymcLib.run_mcmc([prior_cmb_2015_omolb,data]     , variables=variables,niter=niter,nburn=nburn,w_ll_model=' .... put your model name .... ')



