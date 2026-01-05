function eigen_result = common_eig(covmat, options)
    % This function computes the common eigenvectors and eigenvalues of two covariance matrices. Each page of covmat is a covariance matrix.
    % The two matrices must have the same size. This codes uses Manopt toolbox to compute the common eigenvectors.

    % Options:
    % manopt options: the options for the manopt algorithm. The default options are used if not given. 
    % cpc_num: the number of principal components to keep. If not given, use the smaller rank of the input matrices.
    % sample_n: When the input is sample covariance matrices, this option specifies the number of observations to estimate the covariance matrices for each group. If not given, same number of observations is assumed. 

    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end

    if ~isfield(options, 'cpc_num')
        options.cpc_num = nan;
    end

    if ~isfield(options, 'sample_n')
        options.sample_n = ones(1,size(covmat,3)); % set default sample size to 1 for all groups
    end

    if ~isfield(options, 'manopt')
        options.manopt = struct();
    end

    condn = size(covmat,3); % number of conditions

    % first we do eigenvalue decomposition separately to judge whether we could
    % find common eigenvectors

    v = zeros(size(covmat,1),size(covmat,1),condn); % eigenvectors
    d = zeros(size(covmat,1),condn); % eigenvalues
    r = zeros(1,condn); % rank of the covariance matrix
    for i = 1:condn
        [v(:,:,i),tempd] = eig(covmat(:,:,i));
        r(i) = rank(covmat(:,:,i));
        [d(:,i),idx] = sort(diag(tempd),'descend');
        v(:,:,i) = v(:,idx,i);
    end

    % calculate the inner product (angle) between the eigenvectors
    % eig_sim = v1' * v2;
    % figure;
    % subplot(1,3,1); stem(d1(1:r1));title('eigenvalues of covmat 1');
    % subplot(1,3,2); stem(d2(1:r2));title('eigenvalues of covmat 2');
    % subplot(1,3,3); imagesc(eig_sim(1:r1,1:r2));clim([0 1]); title('Inner product between the eigenvectors'); % visually identify common vectors
    % drawnow;
    
    % after confirming 
    p = size(covmat,1); r = min(r); % only consider the top r eigenvectors to reduce computation time

    % Create the problem structure.
    elements.Q = stiefelfactory(p, r, 1);
    elements.D = positivefactory(r,condn);
    manifold = productmanifold(elements);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost = @(x) cost_func(x, covmat, options.sample_n);
    problem.egrad = @(x) egrad_func(x, covmat, options.sample_n);

    % Numerically check gradient consistency (just once, optional).
    % checkgradient(problem); %pause;

    % use the eigenvector of the mean covariance matrix as initial point
    if ~isfield(options, 'init_point')
        [v_avg,d_avg] = eig(mean(covmat,3));
        [d_avg,idx] = sort(diag(d_avg),'descend');
        v_avg = v_avg(:,idx);
        options.init_point.Q = v_avg(:,1:r);
        options.init_point.D = repmat(sqrt(d_avg(1:r)),1,condn);
    end

    % Solve.
    [eigen_result, cost, info, options] = trustregions(problem, options.init_point, options.manopt);

    % sort the common factors by the average eigenvalues
    [~,idx] = sort(mean(eigen_result.D.^2,2),'descend');
    eigen_result.Q = eigen_result.Q(:,idx);
    eigen_result.D = eigen_result.D(idx,:);

end


function output = cost_func(param, covmat, sample_n)
    % this function is to calculate the cost of the problem.

    condn = size(covmat,3);
    output = arrayfun(@(x)(sample_n(x) * norm(covmat(:,:,x) - (param.Q .* (param.D(:,x).^2)') * param.Q',"fro")^2),1:condn);
    output = 1/4 * sum(output);

end

function [egrad] = egrad_func(param, covmat, sample_n)
    % this function is to calculate the gradient of the cost function.

    condn = size(covmat,3);

    temp_q = zeros([size(param.Q),condn]); 
    temp_d = zeros(size(param.D)); 
    for i = 1:condn
        temp_mat = (param.Q .* (param.D(:,i).^2)');

        % egrad for Q
        temp_q(:,:,i) = - sample_n(i) * (covmat(:,:,i)*temp_mat - temp_mat * param.Q' * covmat(:,:,i)*temp_mat);

        % egrad for D
        temp_d(:,i) = - sample_n(i)*param.D(:,i).*(diag(param.Q' * covmat(:,:,i) * param.Q) - param.D(:,i).^2);
    end


    egrad = struct(...
        'Q',sum(temp_q,3),...
        'D', temp_d);

end