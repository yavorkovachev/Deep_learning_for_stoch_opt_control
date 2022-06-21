    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:07:18 2022

@author: Yavor Kovachev
"""

## Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

#matplotlib.use('TkAgg') # Required to make it run on both Windows and Mac
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

### Other libraries 
import numpy as np
import tensorflow as tf

############################################### ###############################
#################           Computing functions        ########################
###############################################################################

def computeRiskPrices(valueFunctionLogH, valueFunctionLogE, policyFunctionKappa, X, params, order_states,
             logXiE = None, logXiH = None, kappa = None, dX_LogXiE = None, dX_LogXiH = None, dX2_LogXiE = None, 
             dX2_LogXiH = None, dX_Q = None, dX2_Q_noncross = None, dX_logQ = None, dX2_logQ = None, typeOfOutput = 'MFR'):
  
  ## valueFunctionH, valueFunctionE:              neural nets used to approximate value functions
  ## policyFunctionLogQ, policyFunctionKappa:     neural nets used to approximate policy functions
  ## X:                                           sampled states
  ## params:                                      parameters

  ## Parse information
  nShocks      = params['sigmaK'].shape[0]
  nStates      = X.shape[1]
  batchSize    = X.shape[0]

  if typeOfOutput == 'NN':  ### NN case
    logXiE       = valueFunctionLogE(X)
    logXiH       = valueFunctionLogH(X)
    xiE          = tf.exp(logXiE)
    xiH          = tf.exp(logXiH)
    kappa        = policyFunctionKappa(X)

    if params['a_h'] > 0:
      kappa      = kappa
    else:
      kappa      = tf.ones([batchSize,1])

    ## Derivatives
    with tf.GradientTape(persistent=True) as t1:
      LogXiE     = valueFunctionLogE(X)
      LogXiH     = valueFunctionLogH(X)
      num_q          = (1 - kappa) * params['a_h'] + kappa * params['a_e'] + 1 / params['phi']
      den_q          = (1 - X[:,order_states['W'], tf.newaxis]) * tf.pow(params['rho_h'], 1 / params['psi_h']) \
      * tf.pow(xiH, 1 - 1 / params['psi_h']) + X[:,order_states['W'], tf.newaxis] * tf.pow(params['rho_e'], 1 / params['psi_e']) \
      * tf.pow(xiE, 1 - 1 / params['psi_e']) + 1 / params['phi']
      Q              = num_q / den_q     ##### eq. (44)
      logQ           = tf.math.log(Q)

    ### First ord. der.
    dX_LogXiE = t1.gradient(LogXiE, X)
    dX_LogXiH = t1.gradient(LogXiH, X)
    dX_logQ   = t1.gradient(logQ, X)

  else:                     ### MRF case
    logXiE       = tf.reshape(logXiE, [batchSize, 1] )
    logXiH       = tf.reshape(logXiH, [batchSize, 1] )
    kappa        = tf.reshape(tf.convert_to_tensor(kappa, dtype=tf.float64 ), [batchSize, 1] )
    # ## Derivatives
    dX_logQ                  = tf.convert_to_tensor(dX_logQ, dtype=tf.float64)
    # dX_LogXiE                = tf.reshape(dX_LogXiE, [batchSize, 1] )
    dX_LogXiH                = tf.convert_to_tensor(dX_LogXiH, dtype=tf.float64)

  dZ_logQ = dX_logQ[:,1:2]
  dV_logQ = dX_logQ[:,2:3]

  ## Compute drifts and volatilities. For now, assume no idio vol (\varsigma = 0 everywhere)
  sigmaK       = params['sigmaK'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
  sigmaZ       = params['sigmaZ'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
  sigmaV       = params['sigmaV'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
  
  ## Compute chi
  sigmaXtilde  = [sigmaZ, sigmaV]
  #### Terms for chi
  Dx           = sigmaK + (sigmaZ*dZ_logQ + sigmaV*dV_logQ)       ###### eq. (68)
  DxNormSq     = tf.reduce_sum(Dx * Dx, axis = 1, keepdims=True)
  DzetaOmega   = DxNormSq * ( ( params['gamma_h'] - 1.0) * dX_LogXiH[:,order_states['W'], tf.newaxis] - 
                                                                    (params['gamma_e'] - 1.0) * dX_LogXiE[:,order_states['W'], tf.newaxis] );

  DzetaOmega   = DzetaOmega * X[:,order_states['W'], tf.newaxis]  * (1 - X[:,order_states['W'], tf.newaxis]) 
  DzetaX       = tf.zeros(DzetaOmega.shape, dtype=tf.float64)

  for s in range(nShocks):
    for n in range(1,nStates):
      DzetaX       = DzetaX + Dx[:,s, tf.newaxis] * ( sigmaXtilde[n-1][:,s, tf.newaxis] * ( ( params['gamma_h'] - 1.0) * dX_LogXiH[:,n, tf.newaxis] - 
                                                                    (params['gamma_e'] - 1.0) * dX_LogXiE[:,n, tf.newaxis] ) ) 
  
  DzetaX       = DzetaX * X[:,order_states['W'], tf.newaxis] * (1 - X[:,order_states['W'], tf.newaxis]) ###### eq. (68)

  # #### Find chi
  chiN         = DzetaX - X[:,order_states['W'], tf.newaxis] * (1 - X[:,order_states['W'], tf.newaxis]) * (params['gamma_e'] - params['gamma_h']) * DxNormSq
  chiD         = ( ( 1 - X[:,order_states['W'], tf.newaxis]) * params['gamma_e'] + X[:,order_states['W'], tf.newaxis] * params['gamma_h'] ) * DxNormSq + dX_logQ[:,0, tf.newaxis] * DzetaX - DzetaOmega
  chi          = chiN / chiD + X[:,order_states['W'], tf.newaxis]  ###### eq. (68)
  chi          = tf.math.maximum(chi, params['chiUnderline'])

  ## Compute deltaE and deltaH
  sigmaQ       = ( (chi * kappa - tf.reshape(X[:, order_states['W']],[batchSize, 1]) ) * sigmaK * tf.reshape(dX_logQ[:,0], [batchSize,1]) + sigmaZ * tf.reshape(dX_logQ[:,1], [batchSize, 1])
   + sigmaV * tf.reshape(dX_logQ[:,2], [batchSize,1] ) )  / (1.0 -  (chi * kappa - tf.reshape(X[:,order_states['W']],[batchSize, 1]) ) * tf.reshape(dX_logQ[:,0], [batchSize,1]) ) ###### eq. (57)

  sigmaR       = sigmaK  + sigmaQ  ###### eq. (58) simplified

  # sigmaRNormSq = tf.reshape(tf.reduce_sum(sigmaR * sigmaR, axis = 1), [batchSize,1]) #tf.reshape( tf.square(tf.norm(sigmaR,ord='euclidean', axis= 1)), [batchSize,1])


  sigmaW       = (chi * kappa - tf.reshape(X[:,order_states['W']], [batchSize,1]) ) * sigmaR ###### eq. (52)

  Pi           = params['gamma_h'] * ( (1.0 - chi * kappa) / (1.0 - tf.reshape(X[:,order_states['W']], [batchSize,1]) )  ) * sigmaR + \
  (params['gamma_h'] - 1.0) * (sigmaW * tf.reshape(dX_LogXiH[:,0], [batchSize, 1]) + sigmaZ * tf.reshape(dX_LogXiH[:,1], [batchSize, 1]) + \
                                      sigmaV * tf.reshape(dX_LogXiH[:,2], [batchSize, 1]) )  ###### eq. (62)

  # Pi = tf.ones(shape=(batchSize,1), dtype=tf.float64)
  return Pi

############################################### ###############################
#################           Plotting functions        #########################
###############################################################################

def plotLosses(totalLosses, lossesHJBE, lossesHJBH, lossesKappa):
  '''
  Plot training losses: total, HJB_e, HJB_h, kappa, and any additional losses 
  from e.g. active training points etc. 
  '''
  plt.rcParams["figure.figsize"] = [12,10]
  fig, axs = plt.subplots(2, 2)
  fig.suptitle('Training losses (errors)', y=0.92)

  axs[0,0].plot(totalLosses)
  axs[0,0].set_title('Total loss')
  axs[0,0].set_xlabel('Number of epochs')
  axs[0,0].set_ylabel('Loss')
  axs[0,0].set_xscale('log')
  axs[0,0].set_yscale('log')

  # HJB experts
  axs[0,1].plot(lossesHJBE)
  axs[0,1].set_title('HJBE experts loss')
  axs[0,1].set_xlabel('Number of epochs')
  axs[0,1].set_ylabel('Loss')
  axs[0,1].set_xscale('log')
  axs[0,1].set_yscale('log')

  # HJB households
  axs[1,0].plot(lossesHJBH)
  axs[1,0].set_title('HJBE households loss')
  axs[1,0].set_xlabel('Number of epochs')
  axs[1,0].set_ylabel('Loss')
  axs[1,0].set_xscale('log')
  axs[1,0].set_yscale('log')

  # Kappa
  axs[1,1].plot(lossesKappa)
  axs[1,1].set_title(r'Policy function ($\kappa$) loss')
  axs[1,1].set_xlabel('Number of epochs')
  axs[1,1].set_ylabel('Loss')
  axs[1,1].set_xscale('log')
  axs[1,1].set_yscale('log')

  # plt.tight_layout(pad=0.5)
  plt.show()

def generate2DPlots(mfr_Results, nn_Results, two_Norms, abs_Diffs):

  logXiE, logXiH, kappa = mfr_Results[0], mfr_Results[1], mfr_Results[2]
  logXiE_NNs, logXiH_NNs, kappa_NNs = nn_Results[0], nn_Results[1], nn_Results[2]
  twoNormXiE, twoNormXiH, twoNormKappa = two_Norms[0], two_Norms[1], two_Norms[2]
  maxAbsDiffXiE, maxAbsDiffXiH, maxAbsDiffKappa = abs_Diffs[0], abs_Diffs[1], abs_Diffs[2]
  
  ## Format everything to 4 dec. places
  float_formatter  = "{0:.4f}"
  twoNormXiE       = float_formatter.format(twoNormXiE)
  twoNormXiH       = float_formatter.format(twoNormXiH)
  twoNormKappa     = float_formatter.format(twoNormKappa)
  maxAbsDiffXiE    = float_formatter.format(maxAbsDiffXiE)
  maxAbsDiffXiH    = float_formatter.format(maxAbsDiffXiH)
  maxAbsDiffKappa  = float_formatter.format(maxAbsDiffKappa)

  ## Plot  
  plt.rcParams["figure.figsize"] = [15,10]

  fig, axs = plt.subplots(3, 1)
  fig.suptitle('MFR vs NNs - value and policy functions ', y=1)

  axs[0].plot(logXiE_NNs, '.', label='NN', markersize=2)
  axs[0].plot(logXiE, '.', label='MFR', markersize=2)
  axs[0].set_title(r'Value function experts $\log\xi_{E}$.  ||MFR - NN||$_2$ = ' + twoNormXiE + ' over ' + str(logXiE_NNs.shape[0]) + ' points. '
                    + 'Max. abs. diff. = ' + maxAbsDiffXiE)
  axs[0].set_ylabel(r'$\log\xi_{E}$')
  axs[0].set_xlabel(r'$W \times Z \times V = 100 \times 30 \times 30 = 90000$ observations')
  axs[0].legend()

  axs[1].plot(logXiH_NNs, '.', label='NN', markersize=2)
  axs[1].plot(logXiH, '.', label='MFR', markersize=2)
  axs[1].set_title(r'Value function households $\log\xi_{H}$. ||MFR - NN||$_2$ = ' + twoNormXiH + ' over ' + str(logXiH_NNs.shape[0]) + ' points. '
                    + 'Max. abs. diff. = ' + maxAbsDiffXiH)
  axs[1].set_ylabel(r'$\log\xi_{H}$')
  axs[1].set_xlabel(r'$W \times Z \times V = 100 \times 30 \times 30 = 90000$ observations')
  axs[1].legend()

  axs[2].plot(kappa_NNs, 'o', label='NN', markersize=2)
  axs[2].plot(kappa, 'o', label='MFR', markersize=2)
  axs[2].set_title(r'Policy function $\kappa$. ||MFR - NN||$_2$ = ' + twoNormKappa + ' over ' + str(kappa_NNs.shape[0]) + ' points. '
                    + 'Max. abs. diff. = ' + maxAbsDiffKappa)
  axs[2].set_ylabel(r'$\kappa$')
  axs[2].set_xlabel(r'$W \times Z \times V = 100 \times 30 \times 30 = 90000$ observations')
  axs[2].legend()

  plt.tight_layout(pad=3.0)
  plt.show()

  
def generateSurfacePlots(mfr_Results, nn_Results, fixed_points, X):
    ### Surface plots of value and policy functions 
    '''
    Generates surface plots of the two value funtions and policy function. 
    
    Parameters:
    mfr_Results  (list): List of len 3 containing the MFR approximations for the
    experts VF, households VF and policy function kappa
    nn_Results   (list): List of len 3 containing the NN approximations for the 
    experts VF, households VF and policy function kappa
    fixed_points (list): List of len 3 with values at which to fix Z or V for 
    when plotting the surfaces for the VFs and policy function
    
    Returns:
    None: Generates a 1x3 plot of surfaces
    '''
    W, Z, V = X[:,0], X[:,1], X[:,2]
    logXiE, logXiH, kappa = mfr_Results[0], mfr_Results[1], mfr_Results[2]
    logXiE_NNs, logXiH_NNs, kappa_NNs = nn_Results[0], nn_Results[1], nn_Results[2] 
    ## Fix Z and V to plot value and policy functions as surfaces
    idxZ_E  = X[:,1] == np.unique(Z)[fixed_points[0]]
    idxZ_H = X[:,1] == np.unique(Z)[fixed_points[1]] 
    idxV_K = X[:,2] == np.unique(V)[fixed_points[2]]
    
    VF_E_val = np.unique(Z)[fixed_points[0]]
    VF_H_val = np.unique(Z)[fixed_points[1]]
    Pol_val  = np.unique(V)[fixed_points[2]]
    
    ## Grid size for W
    n_points = np.unique(W).shape[0]
    
    ## Two norms 
    twoNormXiE_surf = np.linalg.norm(logXiE_NNs[idxZ_E] - logXiE[idxZ_E])
    twoNormXiH_surf = np.linalg.norm(logXiH_NNs[idxZ_H] - logXiH[idxZ_H])
    twoNormKappa_surf = np.linalg.norm(kappa_NNs[idxV_K] - kappa[idxV_K])
    ## Format everything to 4 dec. places
    float_formatter   = "{0:.4f}"
    twoNormXiE_surf   = float_formatter.format(twoNormXiE_surf)
    twoNormXiH_surf   = float_formatter.format(twoNormXiH_surf)
    twoNormKappa_surf = float_formatter.format(twoNormKappa_surf)
    VF_E_val          = float_formatter.format(VF_E_val)
    VF_H_val          = float_formatter.format(VF_H_val)
    Pol_val           = float_formatter.format(Pol_val)
    
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=.05, vertical_spacing=.05,
        subplot_titles=('Experts value function <br> Z fixed at '+ VF_E_val +'<br>  ||diff.||_2 = ' + str(twoNormXiE_surf),
                        'Households value function <br> Z fixed at '+ VF_H_val +'<br>  ||diff.||_2 = ' + str(twoNormXiH_surf),
                        'Kappa policy function <br> V fixed at '+ Pol_val +'<br>  ||diff.||_2 = ' + str(twoNormKappa_surf)),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
    fig.update_layout(
        title='MRF vs NN solutions - surface plots',
        scene =dict(xaxis_title='W', yaxis_title='V', zaxis_title='xi_e'),
        scene2=dict(xaxis_title='W', yaxis_title='V', zaxis_title='xi_h'),
        scene3=dict(xaxis_title='W', yaxis_title='Z', zaxis_title='kappa'),
        title_x = 0.5,
        title_y = 0.98)
    
    ## Experts value function as function of (W, V) at Z=Z[14]
    fig.add_trace(go.Surface(
        x=W[idxZ_E].reshape([n_points, 30], order='F'),
        y=V[idxZ_E].reshape([n_points, 30], order='F'),
        z=logXiE[idxZ_E].reshape([n_points, 30], order='F'),
        colorscale='Viridis', showscale=False, name='MFR', showlegend=True), row=1, col=1)
    fig.add_trace(go.Surface(
        x=W[idxZ_E].reshape([n_points, 30], order='F'),
        y=V[idxZ_E].reshape([n_points, 30], order='F'),
        z=logXiE_NNs[idxZ_E].reshape([n_points,30], order='F'),
        showscale=False, name='NN', showlegend=True), row=1, col=1)
    
    ## Households value function as function of (W, V) at Z=Z[14]
    fig.add_trace(go.Surface(
        x=W[idxZ_H].reshape([n_points, 30], order='F'),
        y=V[idxZ_H].reshape([n_points, 30], order='F'),
        z=logXiH[idxZ_H].reshape([n_points, 30], order='F'),
        colorscale='Viridis', showscale=False), row=1, col=2)
    fig.add_trace(go.Surface(
        x=W[idxZ_H].reshape([n_points, 30], order='F'),
        y=V[idxZ_H].reshape([n_points, 30], order='F'),
        z=logXiH_NNs[idxZ_H].reshape([n_points,30], order='F'),
        showscale=False), row=1, col=2)
    
    ## Policy function kappa as function of (W, Z) at Z=Z[14]
    fig.add_trace(go.Surface(
        x=W[idxV_K].reshape([n_points, 30], order='F'),
        y=Z[idxV_K].reshape([n_points, 30], order='F'),
        z=kappa[idxV_K].reshape([n_points, 30], order='F'),
        colorscale='Viridis', showscale=False), row=1, col=3)
    fig.add_trace(go.Surface(
        x=W[idxV_K].reshape([n_points, 30], order='F'),
        y=Z[idxV_K].reshape([n_points, 30], order='F'),
        z=kappa_NNs[idxV_K].reshape([n_points,30], order='F'),
        showscale=False), row=1, col=3)
    
    fig.show()
  
def plotRiskPrices(Pi_MFR, Pi_NN, X, fix_V_location, fix_Z_location):

    # Plot risk prices Pi with plotly
    Pi_MFR = Pi_MFR.numpy()
    Pi_NN  = Pi_NN.numpy()
    
    n_points = 100
    float_formatter   = "{0:.4f}"
    
    W, Z, V = X[:,0], X[:,1], X[:,2]
    ## Fix V
    idxV     = X[:,2] == np.unique(V)[fix_V_location]
    V_fixed = float_formatter.format(np.unique(V)[fix_V_location])
    ## Fix Z
    idxZ     = X[:,1] == np.unique(Z)[fix_Z_location]
    Z_fixed = float_formatter.format(np.unique(Z)[fix_Z_location])
    
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=.05, vertical_spacing=.05,
        subplot_titles=(r'Risk price (households): first shock; V fixed at '+ V_fixed, #+'<br>  ||diff.||_2 = ' + str(twoNormXiE_surf),
                        r'Risk price (households): first shock; Z fixed at '+ Z_fixed,), #+'<br>  ||diff.||_2 = ' + str(twoNormXiH_surf),),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
    
    fig.update_layout(
        title='MRF vs NN solutions - surface plots',
        scene =dict(xaxis_title='Z', yaxis_title='W', zaxis_title='Risk price 1'),
        scene2=dict(xaxis_title='V', yaxis_title='W', zaxis_title='Risk price 1'),
        title_x = 0.5,
        title_y = 0.98)
    
    ### Risk prices Pi eq. (62). Risk price 1 as a function of (Z, W) with V fixed
    fig.add_trace(go.Surface(
        x=Z[idxV].reshape([n_points, 30], order='F'),
        y=W[idxV].reshape([n_points, 30], order='F'),
        z=Pi_MFR[:,0][idxV].reshape([n_points, 30], order='F'),
        colorscale='Viridis', showscale=False, name='MFR', showlegend=True), row=1, col=1)
    fig.add_trace(go.Surface(
        x=Z[idxV].reshape([n_points, 30], order='F'),
        y=W[idxV].reshape([n_points, 30], order='F'),
        z=Pi_NN[:,0][idxV].reshape([n_points, 30], order='F'),
        showscale=False, name='NN', showlegend=True), row=1, col=1)
    
    ### Risk prices Pi eq. (62). Risk price 1 as a function of (V, W) with Z fixed
    fig.add_trace(go.Surface(
        x=V[idxZ].reshape([n_points, 30], order='F'),
        y=W[idxZ].reshape([n_points, 30], order='F'),
        z=Pi_MFR[:,0][idxZ].reshape([n_points, 30], order='F'),
        colorscale='Viridis', showscale=False), row=1, col=2)
    fig.add_trace(go.Surface(
        x=V[idxZ].reshape([n_points, 30], order='F'),
        y=W[idxZ].reshape([n_points, 30], order='F'),
        z=Pi_NN[:,0][idxZ].reshape([n_points, 30], order='F'),
        showscale=False), row=1, col=2)
    
    # fig.update_layout(height=400, width=1200)
    
    fig.show()
  
def plotRiskPricesNN(valueFunctionLogE, valueFunctionLogH, policyFunctionKappa, params, n_points):
    
    '''
    Generates surface plot risk prices when MFR solutions are not avialable. Grid size is the same in each state variable. 
    
    Parameters:
    valueFunctionLogE: a keras sequential nueral network object
    valueFunctionLogH: a keras sequential nueral network object
    policyFunctionKappa: a keras sequential nueral network object
    params: economic model parameters e.g. risk aversions, IES etc.
    n_points: how many points to use in the grid of each state variable for plotting
    
    Returns: None. Generates a 1x2 plot of surfaces
    '''
    order_states           = {'W':0, 'Z': 1, 'V': 2}
    
    Wt      = tf.reshape(tf.linspace(start = params['wMin'], stop = params['wMax'], num=n_points), shape=(n_points,1))
    Zt      = tf.reshape(tf.linspace(start = params['zMin'], stop = params['zMax'], num=n_points), shape=(n_points,1))
    Vt      = tf.reshape(tf.linspace(start = params['vMin'], stop = params['vMax'], num=n_points), shape=(n_points,1))
      
    Wm, Zm, Vm= np.meshgrid(Wt.numpy(), Zt.numpy(), Vt.numpy(), indexing='ij')
      
    X = np.stack((Wm.flatten(), Zm.flatten(), Vm.flatten()), axis=1)
    X = tf.Variable(X)
      
    nShocks      = params['sigmaK'].shape[0]
    nStates      = X.shape[1]
    batchSize    = X.shape[0]
      
    logXiE       = valueFunctionLogE(X)
    logXiH       = valueFunctionLogH(X)
    xiE          = tf.exp(logXiE)
    xiH          = tf.exp(logXiH)
    kappa        = policyFunctionKappa(X)
      
    if params['a_h'] > 0:
      kappa      = kappa
    else:
      kappa      = tf.ones([batchSize,1])
      
    ## Derivatives
    with tf.GradientTape(persistent=True) as t1:
      LogXiE     = valueFunctionLogE(X)
      LogXiH     = valueFunctionLogH(X)
      num_q          = (1 - kappa) * params['a_h'] + kappa * params['a_e'] + 1 / params['phi']
      den_q          = (1 - X[:,order_states['W'], tf.newaxis]) * tf.pow(params['rho_h'], 1 / params['psi_h']) \
      * tf.pow(xiH, 1 - 1 / params['psi_h']) + X[:,order_states['W'], tf.newaxis] * tf.pow(params['rho_e'], 1 / params['psi_e']) \
      * tf.pow(xiE, 1 - 1 / params['psi_e']) + 1 / params['phi']
      Q              = num_q / den_q     ##### eq. (44)
      logQ           = tf.math.log(Q)
      
    ### First ord. der.
    dX_LogXiE = t1.gradient(LogXiE, X)
    dX_LogXiH = t1.gradient(LogXiH, X)
    dX_logQ   = t1.gradient(logQ, X)
      
    dZ_logQ = dX_logQ[:,1:2]
    dV_logQ = dX_logQ[:,2:3]
      
    ## Compute drifts and volatilities. For now, assume no idio vol (\varsigma = 0 everywhere)
    sigmaK       = params['sigmaK'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
    sigmaZ       = params['sigmaZ'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
    sigmaV       = params['sigmaV'] * tf.sqrt(tf.reshape(X[:,order_states['V']],[batchSize, 1]))
    
    ## Compute chi
    sigmaXtilde  = [sigmaZ, sigmaV]
    #### Terms for chi
    Dx           = sigmaK + (sigmaZ*dZ_logQ + sigmaV*dV_logQ)       ###### eq. (68)
    DxNormSq     = tf.reduce_sum(Dx * Dx, axis = 1, keepdims=True)
    DzetaOmega   = DxNormSq * ( ( params['gamma_h'] - 1.0) * dX_LogXiH[:,order_states['W'], tf.newaxis] - 
                                                                      (params['gamma_e'] - 1.0) * dX_LogXiE[:,order_states['W'], tf.newaxis] );
      
    DzetaOmega   = DzetaOmega * X[:,order_states['W'], tf.newaxis]  * (1 - X[:,order_states['W'], tf.newaxis]) 
    DzetaX       = tf.zeros(DzetaOmega.shape, dtype=tf.float64)
      
    for s in range(nShocks):
      for n in range(1,nStates):
        DzetaX       = DzetaX + Dx[:,s, tf.newaxis] * ( sigmaXtilde[n-1][:,s, tf.newaxis] * ( ( params['gamma_h'] - 1.0) * dX_LogXiH[:,n, tf.newaxis] - 
                                                                      (params['gamma_e'] - 1.0) * dX_LogXiE[:,n, tf.newaxis] ) ) 
    
    DzetaX       = DzetaX * X[:,order_states['W'], tf.newaxis] * (1 - X[:,order_states['W'], tf.newaxis]) ###### eq. (68)
      
    # #### Find chi
    chiN         = DzetaX - X[:,order_states['W'], tf.newaxis] * (1 - X[:,order_states['W'], tf.newaxis]) * (params['gamma_e'] - params['gamma_h']) * DxNormSq
    chiD         = ( ( 1 - X[:,order_states['W'], tf.newaxis]) * params['gamma_e'] + X[:,order_states['W'], tf.newaxis] * params['gamma_h'] ) * DxNormSq + dX_logQ[:,0, tf.newaxis] * DzetaX - DzetaOmega
    chi          = chiN / chiD + X[:,order_states['W'], tf.newaxis]  ###### eq. (68)
    chi          = tf.math.maximum(chi, params['chiUnderline'])
      
    ## Compute deltaE and deltaH
    sigmaQ       = ( (chi * kappa - tf.reshape(X[:, order_states['W']],[batchSize, 1]) ) * sigmaK * tf.reshape(dX_logQ[:,0], [batchSize,1]) + sigmaZ * tf.reshape(dX_logQ[:,1], [batchSize, 1])
     + sigmaV * tf.reshape(dX_logQ[:,2], [batchSize,1] ) )  / (1.0 -  (chi * kappa - tf.reshape(X[:,order_states['W']],[batchSize, 1]) ) * tf.reshape(dX_logQ[:,0], [batchSize,1]) ) ###### eq. (57)
      
    sigmaR       = sigmaK  + sigmaQ  ###### eq. (58) simplified
      
    # sigmaRNormSq = tf.reshape(tf.reduce_sum(sigmaR * sigmaR, axis = 1), [batchSize,1]) #tf.reshape( tf.square(tf.norm(sigmaR,ord='euclidean', axis= 1)), [batchSize,1])
      
      
    sigmaW       = (chi * kappa - tf.reshape(X[:,order_states['W']], [batchSize,1]) ) * sigmaR ###### eq. (52)
      
    Pi           = params['gamma_h'] * ( (1.0 - chi * kappa) / (1.0 - tf.reshape(X[:,order_states['W']], [batchSize,1]) )  ) * sigmaR + \
    (params['gamma_h'] - 1.0) * (sigmaW * tf.reshape(dX_LogXiH[:,0], [batchSize, 1]) + sigmaZ * tf.reshape(dX_LogXiH[:,1], [batchSize, 1]) + \
                                        sigmaV * tf.reshape(dX_LogXiH[:,2], [batchSize, 1]) )  ###### eq. (62)
      
    Pi_1  = np.reshape(Pi[:,0], (n_points,n_points,n_points))
    W_fixed, Z_fixed, V_fixed = n_points//2, n_points//2, n_points//2
      
    fig = make_subplots(
        rows=1, cols=2, 
        horizontal_spacing=.05, vertical_spacing=.05,
        subplot_titles=(r'Risk price (households): first shock <br> V fixed at grid midpoint',
                        r'Risk price (households): first shock <br> Z fixed at grid midpoint',),
        specs=[[{'type': 'surface'}, {'type': 'surface'}]])
      
    fig.update_layout(
        title='Risk prices',
        scene =dict(xaxis_title='Z', yaxis_title='W', zaxis_title='Risk price 1'),
        scene2=dict(xaxis_title='V', yaxis_title='W', zaxis_title='Risk price 1'),
        title_x = 0.5,
        title_y = 0.98)
      
    ### Risk prices Pi eq. (62). Risk price 1 as a function of (Z, W) with V fixed
    fig.add_trace(go.Surface(
        x= Zm[:,:,V_fixed],
        y= Wm[:,:,V_fixed],
        z= Pi_1[:,:,V_fixed],
        colorscale='Viridis', showscale=False, name='NN', showlegend=True), row=1, col=1)
      
    ### Risk prices Pi eq. (62). Risk price 1 as a function of (V, W) with Z fixed
    fig.add_trace(go.Surface(
        x= Vm[:,Z_fixed,:],
        y= Wm[:,Z_fixed,:],
        z= Pi_1[:,Z_fixed,:],
        colorscale='Viridis', showscale=False), row=1, col=2)
      
    # fig.update_layout(height=400, width=1200)
      
    fig.show()

    
## Plotting on a non MFR grid
def generateSurfacePlotsNNs(logXiE_NN, logXiH_NN, kappa_NN, params, n_points):

    Wt      = tf.reshape(tf.linspace(start = params['wMin'], stop = params['wMax'], num=n_points), shape=(n_points,1))
    Zt      = tf.reshape(tf.linspace(start = params['zMin'], stop = params['zMax'], num=n_points), shape=(n_points,1))
    Vt      = tf.reshape(tf.linspace(start = params['vMin'], stop = params['vMax'], num=n_points), shape=(n_points,1))
    
    Wm, Zm, Vm= np.meshgrid(Wt.numpy(), Zt.numpy(), Vt.numpy(), indexing='ij')
    
    inps = np.stack((Wm.flatten(), Zm.flatten(), Vm.flatten()), axis=1)
    
    logXiE_NN_predicted = logXiE_NN(inps)
    logXiH_NN_predicted = logXiH_NN(inps)
    kappa_NN_predicted  = kappa_NN(inps)
    
    logXiE_NN_predicted = np.reshape(logXiE_NN_predicted, (n_points,n_points,n_points))
    logXiH_NN_predicted = np.reshape(logXiH_NN_predicted, (n_points,n_points,n_points))
    kappa_NN_predicted  = np.reshape(kappa_NN_predicted , (n_points,n_points,n_points))
    
    W_fixed, Z_fixed, V_fixed = n_points//2, n_points//2, n_points//2
    
    fig = make_subplots(
        rows=1, cols=3, 
        horizontal_spacing=.01, vertical_spacing=.01,
        subplot_titles=('Experts value funcion <br> Z fixed at grid midpoint', 'Households value function <br> Z fixed at grid midpoint', 'Kappa <br>  V fixed at grid midpoint'),
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])
    fig.update_layout(
        title='Value functions and kappa',
        scene =dict(xaxis_title='W', yaxis_title='V', zaxis_title='xi_e'),
        scene2=dict(xaxis_title='W', yaxis_title='V', zaxis_title='xi_h'),
        scene3=dict(xaxis_title='W', yaxis_title='Z', zaxis_title='kappa', zaxis=dict(range=[0,1.02]) ), ## !!! There is a bug in plotly !!! If kappa is identically eq. to 1 the plot will not render
        title_x = 0.5,
        title_y = 0.98)
    
    # Experts value function as function of (W, V), Z fixed
    fig.add_trace(go.Surface(
        x= Wm[:,Z_fixed,:],
        y= Vm[:,Z_fixed,:],
        z= logXiE_NN_predicted[:,Z_fixed,:],
        colorscale='Viridis', showscale=False, name='NN', showlegend=True), row=1, col=1)
    
    ## Households value function as function of (W, V), Z fixed
    fig.add_trace(go.Surface(
        x= Wm[:,Z_fixed,:],
        y= Vm[ :,Z_fixed,:],
        z= logXiH_NN_predicted[:,Z_fixed,:],
        colorscale='Viridis', showscale=False, name='NN', showlegend=False), row=1, col=2)
    
    ## Kappa as a function of (W, Z), V fixed
    fig.add_trace(go.Surface(
        x= Wm[:,:,V_fixed],
        y= Zm[:,:,V_fixed],
        z= kappa_NN_predicted[:,:,V_fixed],
        colorscale='Viridis', showscale=False, name='NN', showlegend=False), row=1, col=3)
    
    fig.add_trace(go.Surface(
        x= Wm[:,:,V_fixed],
        y= Zm[:,:,V_fixed],
        z= logXiH_NN_predicted[:,Z_fixed,:]-10,
        colorscale='Viridis', showscale=False, name='NN', showlegend=False), row=1, col=3) ### Hardcode things by plotting VF for households shifted down by 10. 
                                                                                            ### I have absolutely no idea how to fix the plotly bug when kappa==1
    fig.show()
    
###############################################################################
#################           Misc helper functions          ####################
###############################################################################  

# def setModelParameters(  nu_newborn = 0.1,  lambda_d      = 0.02,  lambda_Z      = 0.252, lambda_V          = 0.156, lambda_Vtilde     = 1.38,  
#   Z_bar        = 0.0,  V_bar        = 1.0,  delta_e       = 0.05,   delta_h      = 0.05,  a_e               = 0.14,  a_h               = 0.135,
#   rho_e        = 0.05, rho_h        = 0.05, phi           = 3,      gamma_e      = 1,     gamma_h           = 1,     psi_e             = 1, 
#   psi_h        = 1,    sigma_K_norm = 0.04, sigma_Z_norm  = 0.0141, sigma_V_norm = 0.132, sigma_Vtilde_norm = 0.0,   equityIss         = 2, 
#   chiUnderline = 1.0,  alpha_K      = 0.05, delta         = 0.05,   numSds       = 5,     Vtilde_bar        = 0.0,
#   cov11        = 1.0,  cov12        = 0,    cov13         = 0,      cov14        = 0, 
#   cov21        = 0,    cov22        = 1.0,  cov23         = 0,      cov24        = 0, 
#   cov31        = 0,    cov32        = 0,    cov33         = 1.0,    cov34        = 0, 
#   cov41        = 0,    cov42        = 0,    cov43         = 0,      cov44        = 1.0):

#   paramsDefault = {}

#   ####### Model parameters #######
#   paramsDefault['nu_newborn']             = tf.constant(nu_newborn,         dtype=tf.float32);
#   paramsDefault['lambda_d']               = tf.constant(lambda_d,           dtype=tf.float32);
#   paramsDefault['lambda_Z']               = tf.constant(lambda_Z,           dtype=tf.float32);
#   paramsDefault['lambda_V']               = tf.constant(lambda_V,           dtype=tf.float32);
#   paramsDefault['lambda_Vtilde']          = tf.constant(lambda_Vtilde,      dtype=tf.float32);
#   paramsDefault['Vtilde_bar']             = tf.constant(Vtilde_bar,         dtype=tf.float32);
#   paramsDefault['Z_bar']                  = tf.constant(Z_bar,              dtype=tf.float32);
#   paramsDefault['V_bar']                  = tf.constant(V_bar,              dtype=tf.float32);
#   paramsDefault['delta_e']                = tf.constant(delta_e,            dtype=tf.float32);
#   paramsDefault['delta_h']                = tf.constant(delta_h,            dtype=tf.float32);
#   paramsDefault['a_e']                    = tf.constant(a_e,                dtype=tf.float32);
#   paramsDefault['a_h']                    = tf.constant(a_h,                dtype=tf.float32);  ###Any negative number means -infty
#   paramsDefault['rho_e']                  = tf.constant(rho_e,              dtype=tf.float32);
#   paramsDefault['rho_h']                  = tf.constant(rho_h,              dtype=tf.float32);
#   paramsDefault['phi']                    = tf.constant(phi,                dtype=tf.float32);
#   paramsDefault['gamma_e']                = tf.constant(gamma_e,            dtype=tf.float32);
#   paramsDefault['gamma_h']                = tf.constant(gamma_h,            dtype=tf.float32);
#   paramsDefault['psi_e']                  = tf.constant(psi_e,              dtype=tf.float32);
#   paramsDefault['psi_h']                  = tf.constant(psi_h,              dtype=tf.float32);
#   paramsDefault['sigma_K_norm']           = tf.constant(sigma_K_norm,       dtype=tf.float32);
#   paramsDefault['sigma_Z_norm']           = tf.constant(sigma_Z_norm,       dtype=tf.float32);
#   paramsDefault['sigma_V_norm']           = tf.constant(sigma_V_norm,       dtype=tf.float32);
#   paramsDefault['sigma_Vtilde_norm']      = tf.constant(sigma_Vtilde_norm,  dtype=tf.float32);
#   paramsDefault['equityIss']              = tf.constant(equityIss,          dtype=tf.float32);
#   paramsDefault['chiUnderline']           = tf.constant(chiUnderline,       dtype=tf.float32);
#   paramsDefault['alpha_K']                = tf.constant(alpha_K,            dtype=tf.float32);
#   paramsDefault['delta']                  = tf.constant(delta,              dtype=tf.float32);


#   paramsDefault['cov11']                  = tf.constant(cov11,              dtype=tf.float32);
#   paramsDefault['cov12']                  = tf.constant(cov12,              dtype=tf.float32);
#   paramsDefault['cov13']                  = tf.constant(cov13,              dtype=tf.float32);
#   paramsDefault['cov14']                  = tf.constant(cov14,              dtype=tf.float32);

#   paramsDefault['cov21']                  = tf.constant(cov21,              dtype=tf.float32);
#   paramsDefault['cov22']                  = tf.constant(cov22,              dtype=tf.float32);
#   paramsDefault['cov23']                  = tf.constant(cov23,              dtype=tf.float32);
#   paramsDefault['cov24']                  = tf.constant(cov24,              dtype=tf.float32);

#   paramsDefault['cov31']                  = tf.constant(cov31,              dtype=tf.float32);
#   paramsDefault['cov32']                  = tf.constant(cov32,              dtype=tf.float32);
#   paramsDefault['cov33']                  = tf.constant(cov33,              dtype=tf.float32);
#   paramsDefault['cov34']                  = tf.constant(cov34,              dtype=tf.float32);

#   paramsDefault['cov41']                  = tf.constant(cov41,              dtype=tf.float32);
#   paramsDefault['cov42']                  = tf.constant(cov42,              dtype=tf.float32);
#   paramsDefault['cov43']                  = tf.constant(cov43,              dtype=tf.float32);
#   paramsDefault['cov44']                  = tf.constant(cov44,              dtype=tf.float32);

#   paramsDefault['numSds']                 = 5

#   ########### Derived parameters
#   ## Covariance matrices 
#   paramsDefault['sigmaK']                 = tf.concat([paramsDefault['cov11'] * paramsDefault['sigma_K_norm'], 
#                                                       paramsDefault['cov12'] * paramsDefault['sigma_K_norm'],
#                                                       paramsDefault['cov13'] * paramsDefault['sigma_K_norm'],
#                                                       paramsDefault['cov14'] * paramsDefault['sigma_K_norm']], 0)

#   paramsDefault['sigmaZ']                 = tf.concat([paramsDefault['cov21'] * paramsDefault['sigma_Z_norm'], 
#                                                       paramsDefault['cov22'] * paramsDefault['sigma_Z_norm'],
#                                                       paramsDefault['cov23'] * paramsDefault['sigma_Z_norm'],
#                                                       paramsDefault['cov24'] * paramsDefault['sigma_Z_norm']], 0)

#   paramsDefault['sigmaV']                 = tf.concat([paramsDefault['cov31'] * paramsDefault['sigma_V_norm'], 
#                                                       paramsDefault['cov32'] * paramsDefault['sigma_V_norm'],
#                                                       paramsDefault['cov33'] * paramsDefault['sigma_V_norm'],
#                                                       paramsDefault['cov34'] * paramsDefault['sigma_V_norm']], 0)
  
#   ## Min and max of state variables
#   ## min/max for V
#   shape = 2 * paramsDefault['lambda_V'] * paramsDefault['V_bar']  /   (tf.pow(paramsDefault['sigma_V_norm'],2));
#   rate = 2 * paramsDefault['lambda_V'] / (tf.pow(paramsDefault['sigma_V_norm'],2));
#   paramsDefault['vMin'] = 0.00001;
#   paramsDefault['vMax'] = paramsDefault['V_bar'] + paramsDefault['numSds'] * tf.sqrt( shape / tf.pow(rate, 2));

#   ## min/max for Z
#   zVar  = tf.pow(paramsDefault['V_bar'] * paramsDefault['sigma_Z_norm'], 2) / (2 * paramsDefault['lambda_Z'])
#   paramsDefault['zMin'] = paramsDefault['Z_bar'] - paramsDefault['numSds'] * tf.sqrt(zVar)
#   paramsDefault['zMax'] = paramsDefault['Z_bar'] + paramsDefault['numSds'] * tf.sqrt(zVar)

#   ## min/max for W
#   paramsDefault['wMin'] = 0.01
#   paramsDefault['wMax'] = 1 - paramsDefault['wMin'] 

#   return paramsDefault