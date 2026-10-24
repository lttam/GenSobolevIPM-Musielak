% This is the code for the NeurIPS'26 paper:
% "Orlicz–Sobolev with Musielak: An Efficient Regularization Approach for Graph-based IPM"
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

%-------------------------------------------------------------------
% Generalized Sobolev IPM with Musielak Regularization (GSI-M) 
%-------------------------------------------------------------------

% Third-party toolbox
% + Generalized Sobolev transport (GST)

% Using the third-party toolbox to create the graphs G_Log and G_Sqrt
% amazon_OT_1000_RandLLE.mat (G_Log)
% amazon_OT_1000_RandSLE.mat (G_Sqrt)

% For examples:
% -- For building random graph (G_Log / G_Sqrt) from support data points
% (Sobolev transport third-party toolbox)
% + clusteringDataset_buildRandomGraph_Log: build random connected graph G_Log (M
% nodes, and M log(M) edges)
% + clusteringDataset_buildRandomGraph_Sqrt: build random connected graph
% G_Sqrt (M nodes, and M^(3/2) edges)

% Compute generalized Sobolev IPM with Musielak regularization (GSI-M)
% + Compute_GSIPM_EXP1_vec_V5B: compute GSI-M with phi1 = @(X) exp(X) - X - 1;
% + Compute_GSIPM_EXP2_vec_V5B: compute GSI-M with phi2 = @(X) exp(X.^2) - 1;

% Ei_fast_smallpos.m: fast computation for the special exponential integral function
% -- relying on the MATLAB built-in expint function with approximation for
% small positive values.

% Samples
% + test_gsipm1_all.m : for GSI-M with phi1
% + test_gsipm2_all.m : for GSI-M with phi2

% Time consumption for GSI-M with phi1, phi2
% + time_GenSobolevIPM_EXP1_V5B_1T.m : GSI-M-phi1
% + time_GenSobolevIPM_EXP2_V5B_1T.m : GSI-M-phi2

% For the limit case: phi0
% GSI-M is equal to Sobolev transport (ST)

% -- Note:
% + The code uses Graph and Network Algorithms from MATLAB. (e.g., Dijkstra
% algorithm for shortest path from a source point to a destination set of
% points.
% + The code uses Optimization Toolbox from MATLAB. (e.g., fmincon to solve
% the univariate optimization problem for GSI-M).





