# Deep learning for stochastic optimal control

The `Deep_Learning_for_Stoch_Opt_Control.ipynb` notebook in the `Financial_frictions` folder shows how to apply deep neural networks (DNNs) to solve stochastic optimal control problems arising from economic models of financial frictions. Solving such models boils down to finding approximations of high dimensional non-linear PDEs (more commonly referred to as Hamilton-Jacobi-Bellman equations (HJBEs) in the field of stochastic optimal control). In the notebook, these approximations are obtained using deep nueral networks. More precisely, the `Deep_Learning_for_Stoch_Opt_Control.ipynb` notebook illustrates how to solve a 3 dimensional model of financial frictions where the HJBEs (alternatively PDEs) for two types of agents, households indexed by $h$ and experts indexed by $e$, are given by 

$$
\begin{align}
0 = & \frac{\psi_h}{1-\psi_h} \rho_{h}^{1/\psi_h} \xi_{h}^{1-1/\psi_h} - \frac{\rho_h}{1 - \psi_h} + r + \frac{1}{2 \gamma_h}(||\pi||^2 + (\gamma_h \beta_h \sqrt{\varsigma})^2) + \left[\mu_X + \frac{1 - \gamma_h}{\gamma_h} \sigma_X \pi \right] \cdot \partial_X \ln \xi_h \\
& + \frac{1}{2} \left[ tr(\sigma_{X}^{'} \partial_{XX'} \ln \xi_h \sigma_{X}^{}) +  \frac{1 - \gamma_h}{\gamma_h} ||\sigma^{'}_{X} \partial_X \ln \xi_h ||^2 \right]
\end{align}
$$

and

$$
\begin{align}
0 = & \frac{\psi_e}{1-\psi_e} \rho_{e}^{1/\psi_e} \xi_{e}^{1-1/\psi_e} - \frac{\rho_e}{1 - \psi_e} + r + \frac{1}{2 \gamma_e}\frac{(\Delta_e + \pi ⋅ \sigma_{R})^2}{||\sigma_R||^2 + \varsigma}  + \left[ \mu_X + \frac{1 - \gamma_e}{ \gamma_e} \frac{(\Delta_e + \pi ⋅ \sigma_R)}{||\sigma_R||^2 + \varsigma} \sigma_{R} \sigma_{X} \right] \partial_{X} \ln \xi_e \\
& + \frac{1}{2} \left[ tr(\sigma_{X}^{'} \partial_{XX'} \ln \xi_e \sigma_{X}^{}) + \frac{1 - \gamma_e}{\gamma_e}(\sigma_{X}' \partial_{X} \ln \xi_e)' \left[ \gamma_e I_d + (1-\gamma_e) \frac{\sigma_{R} \sigma{R}'}{||\sigma_{R}||^2 + \varsigma} \right] \sigma_{X}' \partial_{X} \ln \xi_e  \right]
\end{align}
$$

An additional complication is posed by the approximation of an ODE embedded in a $\min$ operator. This ODE is used to track the binding constraints in the stochastic optimal control problem from which the PDEs above are derived, and is given by 

$$
\begin{align}
0 = & \min ( 1 - \kappa,  (1-w) \gamma_e \chi \kappa (||\sigma_R||^2 + \varsigma ) - w \gamma_h (1 - \chi \kappa) ||\sigma_R||^2 - w (1-w)\sigma_{R}^{'} \sigma_{X}^{'} \partial_{X} \ln \left( \frac{\xi_{h}^{\gamma_h -1}}{\xi_{e}^{\gamma_e -1} } \right) )
\end{align}
$$

Its solution is also estimated with a deep nueral network. 

The notebook provides further explanations, derivations, code implementation and a comparison with a solution based on finite difference methods.

* **The main result showcased in the notebook is that DNNs approximate the solutions of the two high-dimensional non-linear PDEs (HJBEs) and the ODE above with overall the same level of accuracy as finite difference methods (FMDs) but DNNs are significantly faster: computational time for DNN approximations is 10 minutes and 42 seconds vs 3hours and 42 minutes for FDMs. The DNN approach is more than 20 times faster.**  

## Installing/Replication
The easiest/most convenient way to run the code is to download the `Financial_frictions` folder into your Google drive and run the `Deep_Learning_for_Stoch_Opt_Control.ipynb` notebook on Google `Colab`. Alternatively, if you have access to a GPU and have `TensorFlow` installed you can run on your local machine.
