function all_result = cpca(data1, data2)
    %This function computes the common principal component analysis (CPCA), a generalization of standard PCA. The general aim is to find common principal components of two distinct datasets. When it is assumed that the two datasets collected in different conditions have the same components, but the weights of these components could be different, CPCA can be used to identify the common components and their weights. The algorithm is based on eigenvalue decomposition. 

    % Two datasets data1 and data2 should have the same number of variables. Equal number of observations is not necessary. Each row of data1 and data2 represents an observation, and each column represents a variable.

    % compute the principal components based on the covariance matrix
    options.sample_n = [size(data1,1),size(data2,1)];
    covmat = cat(3,cov(data1),cov(data2));
    eigen_result = common_eig(covmat,options);

    % compute the scores of the data based on the principal components
    all_result.eigenvectors = eigen_result.Q;
    all_result.eigenvalues = [eigen_result.D].^2;

end