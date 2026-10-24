clear all
clc

%==============
% For 1000 run

rng('default')

nRun = 1000;
dim = 100;

w = rand(nRun, dim);
x = rand(nRun, dim);

lg = rand(nRun, dim);
lGG = 1.2;

tic
val = Compute_GSIPM_EXP1_vec_V5B(x, w, lg, lGG)
toc
