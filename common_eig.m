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

    % first we do eigenvalue decomposition separately to judge whether we could
    % find common eigenvectors
    [v1,d1] = eig(covmat(:,:,1));
    [v2,d2] = eig(covmat(:,:,2));
    r1 = rank(covmat(:,:,1)); r2 = rank(covmat(:,:,2));
    [d1,idx] = sort(diag(d1),'descend');
    v1 = v1(:,idx);
    [d2,idx] = sort(diag(d2),'descend');
    v2 = v2(:,idx);

    % calculate the inner product (angle) between the eigenvectors
    eig_sim = v1' * v2;
    figure;
    subplot(1,3,1); stem(d1(1:r1));title('eigenvalues of covmat 1');
    subplot(1,3,2); stem(d2(1:r2));title('eigenvalues of covmat 2');
    subplot(1,3,3); imagesc(eig_sim(1:r1,1:r2));clim([0 1]); title('Inner product between the eigenvectors'); % visually identify common vectors
    drawnow;
    
    % after confirming 
    p = size(covmat,1); r = min([r1,r2]); % only consider the top r eigenvectors to reduce computation time

    % Create the problem structure.
    elements.Q = stiefelfactory(p, r, 1);
    elements.D1 = positivefactory(r,1);
    elements.D2 = positivefactory(r,1);
    manifold = productmanifold(elements);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost = @(x) cost_func(x, covmat, options.sample_n);
    problem.egrad = @(x) egrad_func(x, covmat, options.sample_n);

    % Numerically check gradient consistency (just once, optional).
    %checkgradient(problem); pause;

    % use the eigenvector of the mean covariance matrix as initial point
    [v_avg,d_avg] = eig(mean(covmat,3));
    [d_avg,idx] = sort(diag(d_avg),'descend');
    v_avg = v_avg(:,idx);
    init_point.Q = v_avg(:,1:r);
    init_point.D1 = sqrt(d_avg(1:r));
    init_point.D2 = sqrt(d_avg(1:r));

    % Solve.
    [eigen_result, cost, info, options] = trustregions(problem, init_point, options.manopt);

end


function output = cost_func(param, covmat, sample_n)
    % this function is to calculate the cost of the problem.

    output = sample_n(1) * norm(covmat(:,:,1) - (param.Q .* (param.D1.^2)') * param.Q',"fro")^2 + ...
        sample_n(2) * norm(covmat(:,:,2) - (param.Q .* (param.D2.^2)') * param.Q',"fro")^2;
end

function [egrad] = egrad_func(param, covmat, sample_n)
    % this function is to calculate the gradient of the cost function.
    temp_mat1 = (param.Q .* (param.D1.^2)');
    temp_mat2 = (param.Q .* (param.D2.^2)');
    temp_mat3 = covmat(:,:,1)*temp_mat1;
    temp_mat4 = covmat(:,:,2)*temp_mat2;

    egrad = struct(...
        'Q',-4*sample_n(1)*(temp_mat3-temp_mat1*param.Q'*temp_mat3)-4*sample_n(2)*(temp_mat4-temp_mat2*param.Q'*temp_mat4),...
        'D1', -4*sample_n(1)*param.D1.*(diag(param.Q' * covmat(:,:,1) * param.Q) - param.D1.^2),...
        'D2', -4*sample_n(2)*param.D2.*(diag(param.Q' * covmat(:,:,2) * param.Q) - param.D2.^2));

end