clear all
clc

maxNumCompThreads(1);

typeGG = 'RandLLE'; % log-linear #edges
% typeGG = 'RandSLE'; % sqrt-linear #edges

dsName = 'amazon';
maxKC = 1000;
nSS = 1; % #tree (average for Sobolev)

% DD_SS1, 5, 10, 20
load([dsName '_' num2str(maxKC) '_' typeGG '_Graph.mat']);

% nGG: number of 
randSArray = randperm(nGG);
wwGG = GG.Edges.Weight;

DD_SS = cell(nSS, 1);

runTime_Prep = zeros(nSS, 1);
runTime_Dist = zeros(nSS, 1);

% ================
nPair = 10000;
ff = load([dsName '_ID' num2str(nPair) '.mat']);
% ID: Nx2
ID = ff.ID;
DD_GST = zeros(nPair,1);

for idSS = 1:nSS

    % ------- FOR EACH S0 (randomly choose) ---------
    s0 = randSArray(idSS);
    
    tic
    disp(['...[' num2str(idSS) '] compute the tree path']);
    % tree path!!!
    [trPP, trDD, trEP] = shortestpathtree(GG, s0, 'OutputForm', 'cell');
    
    disp(['...[' num2str(idSS) '] vector representation for each vertex']);
    
    % ---------------
    % ===For GRAPH===
    % vector representation for each vertex 1 --> nGG
    
    disp('......vector representation for each vertex');
    % length(wwGG): #edges in graph GG (can be reduced into #edges in tree)

    vecGG_VV = false(nGG, length(wwGG));
    for ii = 1:nGG % each vertex in graph
        vecGG_VV(ii, trEP{ii}) = true;
    end
     
    % V2: extract ---> TREE
    sumEdgeVal = sum(vecGG_VV, 1);
    idNZ = find(sumEdgeVal>0);
    vecGG_VV_TR = vecGG_VV(:, idNZ); % spare version of vecGG_VV
    wwGG_TR = wwGG(idNZ);

    % (4)
    lGG = sum(wwGG_TR);

    gammaEdge_IndexEdges_TR = false(length(idNZ), length(idNZ));
    for ii = 1:nGG
        tmpEE = trEP{ii};

        tmpEE_idNZ = tmpEE;
        for idII = 1:length(tmpEE)
            tmpEE_idNZ(idII) = find(idNZ == tmpEE(idII));
        end

        for jj = 1:(length(tmpEE_idNZ)-1)
            gammaEdge_IndexEdges_TR(tmpEE_idNZ(jj), tmpEE_idNZ((jj+1):end)) = true;
        end
    end

    gammaEdge_WeightsMatrix_TR = repmat(wwGG_TR', length(idNZ), 1);
    gammaEdge_Vec_TR = sum((gammaEdge_IndexEdges_TR .* gammaEdge_WeightsMatrix_TR), 2);
  
    disp('......vector representation for each distribution');

    XX_SI = zeros(N, length(idNZ));
    
    for ii = 1:N % each distribution
        tmpWW = WW{ii}/sum(WW{ii}); % normalization for weight!!!
        tmpXX = XX_ID{ii};
        
        tmpXX_GG_TR = vecGG_VV_TR(tmpXX, :);
        
        tmpWW_GG_TR = repmat(tmpWW, 1, length(idNZ));

        tmpWWXX = tmpXX_GG_TR .* tmpWW_GG_TR;
        XX_SI(ii, :) = sum(tmpWWXX, 1);
    end
    runTime_Prep(idSS) = toc;
   
    ii = ID(:, 1);
    jj = ID(:, 2);

    tmpII_mat = XX_SI(ii, :);
    tmpJJ_mat = XX_SI(jj, :);
        
    tmpAbsDD_mat = abs(tmpII_mat - tmpJJ_mat);             
    wwGG_TR_mat = repmat(wwGG_TR', nPair, 1);
    gammaEdge_Vec_TR_mat = repmat(gammaEdge_Vec_TR', nPair, 1);

    tic    
    % Generalized Sobolev IPM
    DD_SS_II = Compute_GSIPM_EXP1_vec_V5B(tmpAbsDD_mat, wwGG_TR_mat, gammaEdge_Vec_TR_mat, lGG);
    runTime_Dist(idSS) = toc;
    
    % save distance matrix
    DD_SS{idSS} = DD_SS_II;

end

runTime_Prep_Avg = runTime_Prep(1);
runTime_Dist_Avg = runTime_Dist(1);

runTime_Dist_ALL = runTime_Prep + runTime_Dist;
runTime_Dist_ALL_Avg = runTime_Dist_ALL(1);

DD_SS1 = DD_SS{1};


outName = [dsName '_Time_GenSobolevIPM_Exp1_V5B_' num2str(maxKC) '_' typeGG '_' num2str(nPair) '_S' num2str(nSS) '.mat'];
    
save(outName, 'DD_SS1', ...
     'runTime_Dist', 'runTime_Prep', 'runTime_Dist_ALL', ...
     'runTime_Dist_Avg', 'runTime_Prep_Avg', 'runTime_Dist_ALL_Avg', ...
     'randSArray', 'nSS', 'nPair');


disp('FINISH !!!');



