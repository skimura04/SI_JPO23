#!/usr/bin/env python

# Assess the transient growth of hydrostatic symmetric instability problem
# The code demonstrates
# 1. the transient growth rate converges to the normal mode growth rate at t->\infty (eigenvalue of A)
# 2. the transeint growth rate converges to the initial mode growth rate at t-> 0 (eigenvalue of 0.5*(A+A.H))
# The matrix A is the propagator for the hydrostatic symmetric instability problem.
# This code is used in "Initial and transient growth of symmetric instability", submitted to JPO.
#  
# Author: Satoshi Kimura, Japan Agency for Marine-Earth Science and Technology, Octorber 2023



import sys

import time

from numpy import linspace,zeros,real,imag,sqrt,ones,pi,argsort,array,log,matmul,conj,trapz,matrix,exp,nan,savez
import pylab
from scipy.linalg import svd,expm,det,inv,eig


def Propagator_SymHydroND(rich,Ro):
	### PURPOSE
	## This routine generates the progagator of the normal mode hydrostatic SI problem.
	## The 3x3 matrix in the first part of equation (19) with s = delta/rich, that is along the isopycnal.
	## The eigenvalue of this matrix corresponds to the normal mode growth rate, \lambda in the paper.
	### Inputs
	## 1. rich: Richardson number, Bz/(Uz^2)
	## 2. Ro: Rossby number, -Uy/f
	ii = complex(0.0,1.0)
	A = zeros((3,3))+ii*zeros((3,3))
	A[0,1] = 1+Ro - 1/rich
	A[1,0] = -1
	A[1,2] = 1/sqrt(complex(rich,))
	
	A = matrix(A)
	return A

def Compute_time_derivative(A,u,v,b):
	### PURPOSE
	## Compute the derivative from the propagator matrix, [ut,vt,bt].T = A [u,v,b].T
	### Inputs
	## 1. A: the propagator matrix
	## 2. u: streamwise perturbation velocity
	## 3. v: spanwise perturbation velocity
	## 4. b: buoyancy perturbation

	vec = array([u,v,b]).T # transpose
	svec = matmul(A,vec) # multiply the matrix

	## Here is the time derivative
	ut = svec[0,0]
	vt = svec[0,1]
	bt = svec[0,2]

	return ut,vt,bt

def Compute_SymHydro_eigs(Ro,rich,delta,FGM_type,fmode):
	### PURPOSE
	## Calculate the growth rate, u, v, b along the isopycnal slope, s= delta/rich, from the background flow parameters, Ro,rich, and delta.
	## FGM_type specifies the initial or normal mode growth, FGM_type = 'INST' -> initial mode and FGM_type = 'NORM' -> normal mode.
	### Inputs
	## 1. rich: Richardson number, Bz/(Uz^2)
	## 2. Ro: Rossby number, -Uy/f
	## 3. delta: the ratio of the Coriolis parameter to the thermal wind shear, f/Uz
	## 4. FGM_type: Specifies if the routine computes initial mode or normal mode.
	##	  FGM_type = 'INST' -> initial mode and FGM_type = 'NORM' -> normal mode.
	## 5. fmode: fmode=0 means the fastest growing mode, fmode=1 means the second mode. This problem is 3x3 matrix so fmode<=2

	A = Propagator_SymHydroND(rich,Ro) # Define the propagator matrix

	## equation 18 
	if FGM_type=='INST':
		AA = 0.5*(A+A.H)
	elif FGM_type=='NORM':
		AA = A

	sigs, vecs = eig(AA) ## compute the eigen value

	############## Extract the FGM
	ind = argsort(real(sigs))
	sigs_sorted = sigs[ind[::-1]]
	vecs_sorted = vecs[:,ind[::-1]]

	FF = dict()
	FF['AA'] = AA ## Output the matrix

	FF['sigs_sorted'] = sigs_sorted # sorted eigenvalues, that is the growth rate
	FF['vecs_sorted'] = vecs_sorted # sorted eigenvectors, which consist of u, v, and b.

	### Pick the mode of interest by fmode
	u = vecs_sorted[0,fmode]
	v = vecs_sorted[1,fmode]
	b = vecs_sorted[2,fmode]
	s = sigs_sorted[fmode]

	[u_t,v_t,b_t] = Compute_time_derivative(A,u,v,b) # now compute the time derivative

	FF['ke'] = u*conj(u) + v*conj(v) ## kinetic energy
	FF['pe'] = b*conj(b) ## potential energy
	FF['dke'] = conj(u)*u_t + conj(v)*v_t ## rate of change in the kinetic energy
	FF['dpe'] = conj(b)*b_t # rate of change in the potential energy

	FF['u'] = u # scaled spanwise velocity perturbation
	FF['v'] = v # scaled streamwise velocity perturbation
	FF['b'] = b # scaled buoyancy perturbation

	## Remember we have scaled the progagator in equation 14. The purpose of the scaling is to make sure that the norm of the vector corresponds to the total energy.
	### We now scale back to u,v, and b
	su = u*sqrt(2.0)
	sv = v*sqrt(2.0)

	if rich>0:
		sb = b*sqrt(2.0*rich/delta**2)
	else:
		aa = complex(2.0*rich/delta**2)
		sb = b*sqrt(aa)


	sw = delta*sv/rich # the vertical velocity perturbation (sw), using the continuity equation

	### Scale back the time derivatives
	su_t = u_t*sqrt(2.0)
	sv_t = v_t*sqrt(2.0)
	
	if rich>0:
		sb_t = b_t*sqrt(2.0*rich/delta**2)
	else:
		aa = complex(2.0*rich/delta**2)
		sb_t = b_t*sqrt(aa)

	FF['su'] = su
	FF['sv'] = sv
	FF['sb'] = sb
	FF['sw'] = sw

	FF['ske'] = 0.5*(conj(su)*su + conj(sv)*sv)
	FF['spe'] = 0.5*(conj(sb)*sb*delta**2/abs(rich))

	## the energy terms
	FF['sGSP'] = real(-conj(su)*sw/delta) ## geostrophic shear production
	FF['sLSP'] = real(conj(su)*sv*Ro) ## lateral shear production
	FF['sWB'] = real(delta*sv*conj(sb)/rich) ## meridional buoyancy, which corresponds to the vertical buoyancy flux along the isopycnal.
	FF['eRHS'] = FF['sGSP'] + FF['sLSP'] + FF['sWB'] ## the right-hand-side of the energy equation
	print('FGM_type,',FGM_type,FF['sGSP'],FF['sLSP'],FF['sWB'])
	FF['eLHS'] = 2*s*(FF['ske']+FF['spe']) ## the left-hand-side of the energy evolution (time rate of change in the total energy)
	
	FF['sig'] = s # the growth rate

	d = abs(FF['eLHS']-FF['eRHS'])
	if (real(s)>0.01)&(d>0.01):
		## Reality check!
		## eLHS ＝ eRHS
		print(FF['eLHS'],FF['eRHS'])
		print('GSP',FF['sGSP'])
		print('LSP',FF['sLSP'])
		print('WB',FF['sWB'])
		print('sig, ',FF['sig'])
		print('Energy does not seem to balance so quit')
		sys.exit()
	return FF

def Compute_SymHydro_EVOLUTION_SVD(tt,Ro,rich,delta,fmode):
	### PURPOSE
	## Using the singular value decomposition (SVD) to assess the transient growth
	### INPUTS
	## 1. tt: an array of nondimensional time
	## 2. rich: Richardson number, Bz/(Uz^2)
	## 3. Ro: Rossby number, -Uy/f
	## 4. delta: the ratio of the Coriolis parameter to the thermal wind shear, f/Uz
	## 5. fmode: fmode=0 means the fastest growing mode, fmode=1 means the second mode. This problem is 3x3 matrix so fmode<=2

	rich = complex(rich,0)
	ii = complex(0,1.0) ## complex number
	## Define the propagator
	A = Propagator_SymHydroND(rich,Ro) ## define the propagator

	nt = len(tt)

	## evolved perturbation
	uL = zeros(nt) + ii*zeros(nt) #u
	vL = zeros(nt) + ii*zeros(nt) #v
	bL = zeros(nt) + ii*zeros(nt) #b

	uL_t = zeros(nt) + ii*zeros(nt)
	vL_t = zeros(nt) + ii*zeros(nt)
	bL_t = zeros(nt) + ii*zeros(nt)

	## initial perturbation that evolves to "evolved perturbation"
	uR = zeros(nt) + ii*zeros(nt)
	vR = zeros(nt) + ii*zeros(nt)
	bR = zeros(nt) + ii*zeros(nt)

	uR_t = zeros(nt) + ii*zeros(nt)
	vR_t = zeros(nt) + ii*zeros(nt)
	bR_t = zeros(nt) + ii*zeros(nt)

	ss = zeros(nt)

	## Assess the transient growth for each timestep in "tt".
	for ft in range(0,len(tt),1): 
		loc_time = tt[ft]
		Exponent1 = array(A*loc_time)
		Forward = expm(Exponent1) ## Compute the matrix exponential 
		[U1, s1, Vh1] = svd(Forward) ## Apply the singular value decomposition (SVD), Forward == U1 @ s1 @ Vh1

		### save the singular value and other variables into the arrays
		ss[ft] = s1[fmode] ## singular value of "fmode"
		uL[ft] = U1[0,fmode]#*sqrt(2.0)
		vL[ft] = U1[1,fmode]#*sqrt(2.0)
		bL[ft] = U1[2,fmode]#*sqrt(2.0*rich/delta**2)

		V1 = matrix(Vh1).H
		uR[ft] = V1[0,fmode]#*sqrt(2.0)
		vR[ft] = V1[1,fmode]#*sqrt(2.0)
		bR[ft] = V1[2,fmode]#*sqrt(2.0*rich/delta**2)

		## Compute the time derivative of the evolved perturbation
		[uL_t[ft],vL_t[ft],bL_t[ft]] = Compute_time_derivative(A,uL[ft],vL[ft],bL[ft])

		## Compute the time derivative of the initial perturbation
		[uR_t[ft],vR_t[ft],bR_t[ft]] = Compute_time_derivative(A,uR[ft],vR[ft],bR[ft])

	FF = dict()
	## initial and evolved u
	FF['uL'] = uL
	FF['uR'] = uR
	## initial and evolved v
	FF['vL'] = vL
	FF['vR'] = vR
	## initial and evolved b
	FF['bL'] = bL
	FF['bR'] = bR
	## singular value
	FF['ss'] = ss
	
	## the energy budget (evolved)
	FF['eL'] = conj(uL)*uL + conj(vL)*vL + conj(bL)*bL # energy
	FF['eL_t'] = conj(uL)*uL_t + conj(vL)*vL_t + conj(bL)*bL_t # time rate of change in energy
	FF['sL'] = FF['eL_t']/FF['eL'] # growth rate

	## the energy budget
	FF['eR'] = uR*uR + vR*vR + bR*bR
	FF['eR_t'] = uR*uR_t + vR*vR_t + bR*bR_t
	FF['sR'] = FF['eR_t']/FF['eR']

	### So far we have applied SVD to the scaled matrix. Now we scale back.
	### scaled
	suR = uR*sqrt(2.0)
	svR = vR*sqrt(2.0)
	sbR = bR*sqrt(2.0*rich/delta**2)
	swR = delta*svR/rich
	FF['sGSP_R'] = -conj(suR)*swR/delta # GSP
	FF['sLSP_R'] = conj(suR)*svR*Ro #LSP
	FF['sWB_R'] = conj(swR)*sbR # MB, meridional buoyancy
	FF['sde_R'] = FF['sGSP_R'] + FF['sLSP_R'] + FF['sWB_R']
	FF['seR'] = 0.5*(suR*suR + svR*svR + sbR*sbR*delta**2/abs(rich))
	FF['ssR'] = (FF['sde_R']/FF['seR'])/2.0

	#### scaled
	suL = uL*sqrt(2.0)
	svL = vL*sqrt(2.0)
	sbL = bL*sqrt(2.0*rich/delta**2)
	swL = delta*svL/rich
	FF['sGSP_L'] = -conj(suL)*swL/delta
	FF['sLSP_L'] = conj(suL)*svL*Ro
	FF['sWB_L'] = conj(swL)*sbL
	FF['sde_L'] = FF['sGSP_L'] + FF['sLSP_L'] + FF['sWB_L']
	FF['seL'] = 0.5*(suL*suL + svL*svL + sbL*sbL*delta**2/abs(rich))
	FF['ssL'] = (FF['sde_L']/FF['seL'])/2.0

	FF['tt'] = tt


	return FF

def AnalyticalExpression(Ro,rich,delta):
	#### Purpose
	## Compute the analytical expression of the normal mode (equaiton 24 and 25) and inital mode (equation 32 and 33)
	rich = complex(rich,0)
	### Normal mode
	N = dict()
	N['sig1'] = sqrt(complex(1/rich - Ro - 1,0))
	N['norm1'] = sqrt(N['sig1']**2 + 1)
	N['u1'] = -N['sig1'] #/N['norm1']
	N['v1'] = 1#/N['norm1']
	N['b1'] = 0#/N['norm1']
	N['w1'] = delta*N['v1']/rich

	N['su1'] = N['u1']*sqrt(2.0)
	N['sv1'] = N['v1']*sqrt(2.0)
	N['sb1'] = N['b1']*sqrt(2.0*rich/delta**2)
	N['sw1'] = delta*N['sv1']/rich

	N['sGSP'] = -conj(N['su1'])*N['sw1']/delta
	N['sLSP'] =  conj(N['su1'])*N['sv1']*Ro
	N['sWB'] = delta*N['sv1']*conj(N['sb1'])/rich 
	N['sKE'] = 0.5*(N['su1']*conj(N['su1'])+N['sv1']*conj(N['sv1']) )
	N['sPE'] = 0.5*(N['sb1']*conj(N['sb1']))*delta**2/rich


	## Initial mode
	I = dict()

	I['sig1'] = 0.5*sqrt(complex((1/rich - Ro)**2 + 1/abs(rich),0)) 
	I['norm1'] = sqrt(rich*(-Ro+1/rich)**2 + 4*rich*I['sig1'] +1.0   )
	I['u1'] = conj(sqrt(rich))*(Ro - 1/rich)#/I['norm1']
	I['v1'] =2*conj(sqrt(rich))*I['sig1']#/I['norm1']
	I['b1'] = 1.0#/I['norm1']
	I['w1'] = delta*I['v1']/rich

	I['su1'] = I['u1']*sqrt(2.0)
	I['sv1'] = I['v1']*sqrt(2.0)
	I['sb1'] = I['b1']*sqrt(2.0*rich/delta**2)
	I['sw1'] = delta*I['sv1']/rich

	I['sGSP'] = -conj(I['su1'])*I['sw1']/delta
	I['sLSP'] =  conj(I['su1'])*I['sv1']*Ro
	I['sWB'] = delta*I['sv1']*conj(I['sb1'])/rich 
	I['sKE'] = 0.5*(I['su1']*conj(I['su1']) + I['sv1']*conj(I['sv1']) )
	I['sPE'] = 0.5*(I['sb1']*conj(I['sb1']))*delta**2/abs(rich)

	I['eRHS'] = I['sGSP'] + I['sLSP'] + I['sWB']
	I['eLHS'] = 2*I['sig1']*(I['sKE']+I['sPE'])

	return N,I


##### Select, rich, delta, and Ro
rich = 0.7
delta = 0.1
Ro = -0.5



fmode = 0 # We are interested in the fastest-growing mode
FGM_type = 'NORM'
F = Compute_SymHydro_eigs(Ro,rich,delta,FGM_type,fmode)

FGM_type = 'INST'
I = Compute_SymHydro_eigs(Ro,rich,delta,FGM_type,fmode)

### Call the analytical expression of the normal and initial mode growth rates
[F1,I1] = AnalyticalExpression(Ro,rich,delta)

rich = complex(rich,0)
### The normal mode energy term from rich, delta, and Ro.
iGSP = -2*(abs(rich)/rich)*(Ro-1/rich)*sqrt((1/rich-Ro)**2 +1/abs(rich))
iLSP = 2*(abs(rich)/rich)*Ro*rich*(Ro-1/rich)*sqrt((1/rich-Ro)**2 +1/abs(rich))
iAP = 2*sqrt((1/rich-Ro)**2 +1/abs(rich))
print('GSP comparison',I['sGSP'],I1['sGSP'],iGSP)
print('LSP comparison',I['sLSP'],I1['sLSP'],iLSP)
print('AP comparison',I['sWB'],I1['sWB'],iAP)

#### the initial growth and energy terms
s = sqrt((1/rich-Ro)**2 + 1/abs(rich))
u = conj(sqrt(2*rich))*(Ro-1/rich)
v = conj(sqrt(2*rich))*s
b = sqrt(2*rich/delta**2)
E = 0.5*(conj(u)*u + conj(v)*v + conj(b)*b*delta**2/abs(rich))

iGSP1= -2*(abs(rich)/rich)*(Ro-1/rich)*s #/E
iLSP1 = 2*(abs(rich))*Ro*(Ro-1/rich)*s #/E
iAP1 = 2*s #/E


###########################################
#####
##### START the computation of the transiet growth rate
#####
###########################################
tt = linspace(0,10,1000) # define the nondimensional time.
tt = tt[1::] # Cannot compute the rate at t=0, so we exclude 0.
E = Compute_SymHydro_EVOLUTION_SVD(tt,Ro,rich,delta,fmode) # call the subroutine above

#### save the transient growth rate
runname = './main_energy_evolution_FIG_Ri_'+str(round(real(rich),2))+'_Ro_'+str(round(Ro,2))+'_delta_'+str(round(delta,2))
savefname = runname
## output
G = dict()
G['rich'] = rich
G['Ro'] = Ro
G['delta'] = delta
G['runname'] = runname
G['tt'] = E['tt']
G['ssL'] = E['ssL'] # Transient growth rate, array of len(tt)
G['sig_init'] = I['sig'] # Initial mode growth rate, one value
G['sig_norm'] = F['sig'] # Normal mode growth rate, one value

G['sGSP_L'] = 0.5*E['sGSP_L'] # Transient GSP rate
G['sGSP_init'] = 0.5*I['sGSP'] # initial mode GSP rate
G['sGSP_norm'] = 0.5*F['sGSP'] # normal mode GSP rate

G['sLSP_L'] = 0.5*E['sLSP_L'] # Transient LSP rate
G['sLSP_init'] = 0.5*I['sLSP'] # initial mode LSP rate
G['sLSP_norm'] = 0.5*F['sLSP'] # normal mode LSP rate

G['sAP_L'] = 0.5*E['sWB_L'] # Transient vertical buoyancy flux rate, MB in the paper
G['sAP_init'] = 0.5*I['sWB'] # initial mode MB rate
G['sAP_norm'] = 0.5*F['sWB'] # normal mode MB rate

savez(savefname,**G)

### The relationship that needs to be tested:



fsize = 12
######### plot
figW = 8*2
figH = 11*1.8#11*1.9

msize = 2
lsize = 4
fig1 = pylab.figure(1,figsize=(figW,figH))
params = {'backend': 'ps',
                                'axes.labelsize': fsize,
                                'legend.fontsize': fsize,
                                'xtick.labelsize': fsize,
#                                   'title.fontsize': fsize,
                                'ytick.labelsize': fsize,
                                'font.weight': 'bold'}
pylab.rcParams.update(params)
from matplotlib import rcParams
rcParams['axes.labelpad'] = 10



ax2 = pylab.subplot(4,1,1)
###	1. Does the total energy growth rate coverge to initial and normal mode? 
### G['ssL'][0] ~ I['sig'] at t~0
### G['ssL'][-1] ~ F['sig'] at t~\infty (large t)

ax2.plot(E['tt'],E['ssL'],'ko-',label='Transient growth rate')
ax2.plot(E['tt'],F['sig']*ones(len(E['tt'])),'r-',label='Normal mode growth rate')
ax2.plot(E['tt'],I['sig']*ones(len(E['tt'])),'g-',label='Initial mode growth rate')
ax2.legend(loc="upper right")
ax2.set_title('Energy growth rate',loc='left', fontweight='bold',fontsize=fsize)


ax3 = pylab.subplot(4,1,2)
###	2 Does the GSP growth rate coverge to initial and normal mode? 
### G['sGSP_L'][0] ~ I['sGSP'] at t~0
### G['sGSP_L'][-1] ~ F['sGSP'] at t~\infty (large t)

ax3.plot(E['tt'],0.5*E['sGSP_L'],'ko-',label='Transient GSP growth rate')
ax3.plot(E['tt'],0.5*F['sGSP']*ones(len(E['tt'])),'r-',label='Normal mode GSP growth rate')
ax3.plot(E['tt'],0.5*I['sGSP']*ones(len(E['tt'])),'g-',label='Initial mode GSP growth rate')
ax3.legend(loc="upper right")
ax3.set_title('GSP growth rate',loc='left', fontweight='bold',fontsize=fsize)

ax4 = pylab.subplot(4,1,3)
###	3 Does the LSP growth rate coverge to initial and normal mode? 
### G['sLSP_L'][0] ~ I['sLSP'] at t~0
### G['sLSP_L'][-1] ~ F['sLSP'] at t~\infty (large t)

ax4.plot(E['tt'],0.5*E['sLSP_L'],'ko-',label='Transient LSP growth rate')
ax4.plot(E['tt'],0.5*F['sLSP']*ones(len(E['tt'])),'r-',label='Normal mode LSP growth rate')
ax4.plot(E['tt'],0.5*I['sLSP']*ones(len(E['tt'])),'g-',label='Initial mode LSP growth rate')
ax4.legend(loc="upper right")
ax4.set_title('LSP growth rate',loc='left', fontweight='bold',fontsize=fsize)

ax5 = pylab.subplot(4,1,4)
###	4. Does the MB growth rate coverge to initial and normal mode? 
### G['sWB_L'][0] ~ I['sWB'] at t~0
### G['sWB_L'][-1] ~ F['sWB'] at t~\infty (large t)

ax5.plot(E['tt'],0.5*E['sWB_L'],'ko-',label='Transient MB growth rate')
ax5.plot(E['tt'],0.5*F['sWB']*ones(len(E['tt'])),'r-',label='Normal mode MB growth rate')
ax5.plot(E['tt'],0.5*I['sWB']*ones(len(E['tt'])),'g-',label='Initial mode MB growth rate')
ax5.legend(loc="upper right")
ax5.set_title('MB growth rate',loc='left', fontweight='bold',fontsize=fsize)


##### it works because of the unit energy
LHS = E['ssL']
RHS = 0.5*(E['sGSP_L']+E['sLSP_L']+E['sWB_L'])

sig_imag = max(imag(F['sigs_sorted']))
T = 2*pi/abs(sig_imag)

if sig_imag==0:
	Tmax = max(tt)+1
else:
	Tmax = T+1

ax2.set_xlim((0,Tmax))

ax3.set_xlim((0,Tmax))
ax4.set_xlim((0,Tmax))
ax5.set_xlim((0,Tmax))

ax2.set_ylabel('Nondimensional growth rate', fontweight='bold',fontsize=fsize)
ax3.set_ylabel('Nondimensional growth rate', fontweight='bold',fontsize=fsize)
ax4.set_ylabel('Nondimensional growth rate', fontweight='bold',fontsize=fsize)
ax5.set_ylabel('Nondimensional growth rate', fontweight='bold',fontsize=fsize)

ax5.set_xlabel('Nondimensional time', fontweight='bold',fontsize=fsize)
fig1.savefig('./main_energy_evolution.png')

pylab.show()



