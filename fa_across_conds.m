function result = fa_across_conds(corrmat,options)
    %this function computes the common and unique factors across conditions in the same logic of common principal component analysis. A conventional factor analysis for one correlation matrix follows equation (5.19) in Trendafilov, N., & Gallo, M. (2021), i.e., decomposes R into the sum of QD^2Q^T and Psi^2. Here, we extend this to multiple (two for now) correlation matrices by decomposing R into the sum of QD^2Q^T and Psi^2 for each correlation matrix, assuming the same Q across conditions. This code refers to the codes in page 156-163 in Trendafilov, N., & Gallo, M. (2021).

    % Inputs:
    % Each page of corrmat is a correlation matrix.
    % The two matrices must have the same size. This code uses Manopt toolbox.

    % Options:
    % manopt options: the options for the manopt algorithm. The default options are used if not given. 
    % common_rank: the number of common factors to keep. If not given, use the smaller rank of the input matrices minus 1.
    % sample_n: When the input is sample correlation matrices, this option specifies the number of observations to estimate the correlation matrices for each group. If not given, same number of observations is assumed. 
    % method: ML, LS or GLS method. If not given, use LS.

    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end

    if ~isfield(options, 'common_rank')
        options.common_rank = nan;
    end

    if ~isfield(options, 'sample_n')
        options.sample_n = ones(1,size(corrmat,3)); % set default sample size to 1 for all groups
    end

    if ~isfield(options, 'manopt')
        options.manopt = struct();
    end

    if ~isfield(options, 'method')
        options.method = 'LS';
    end

    p = size(corrmat,1); % number of variables
    r = options.common_rank;

    % set random start point
    [Q,Rw] = qr(rand(p,r)-.5,0); % random start point for Q
    idx = find(diag(Rw)<0); Q(:,idx) = -Q(:,idx); % ensure positive diagonal in Rw
    S.Q = Q; S.D = rand(r,1) - .5; S.Psi = diag(corrmat-Q*diag(S.D.^2)*Q'); % initial parameters

    % create the problem structure
    elements = struct();
    elements.Q = stiefelfactory(p, r);
    elements.D = euclideanfactory(r,1);
    elements.Psi = euclideanfactory(p,1);
    M = productmanifold(elements);
    problem.M = M;

    % define the cost function and its gradient
    problem.costgrad = @(S) costgrad(corrmat, S.Q, S.D, S.Psi, options.method);

    % Numerically check gradient consistency (just once, optional).
    checkgradient(problem); pause;

    % Now solve 
    init_point = S;
    [result, cost, info] = trustregions(problem, init_point, options.manopt);


    % display some stats
    figure;
    if strcmp(options.method, 'ML')
        plot([info.iter], [info.gradnorm], '.-'); hold on;
        plot([info.iter], [info.cost], 'ro-');
    else 
        semilogy([info.iter], [info.gradnorm], '.-'); hold on;
        semilogy([info.iter], [info.cost], 'ro-');
    end
    xlabel('Iteration');
    ylabel('Gradient norm');


end

function [f, G] = costgrad(R,Q,D,Psi,method)
    [p,r] = size(Q);
    D2 = D.^2;
    Psi2 = Psi.^2;
    QD = Q.*repmat(D',p,1);
    RM = QD*QD' + diag(Psi2);

    Y = R - RM;

    % objective function
    switch method
        case 'ML'
            Y = (RM\Y)/RM;
            f = (log(det(RM)) + trace(RM\R))/2;
        case 'GLS'
            W = Y/R;
            Y = R\W;
            f = trace(W*W)/4;
        case 'LS'
            f = trace(Y'*Y)/4;
        otherwise
            error('Unknown method');
    end


    % gradient
    YQ = Y * Q;
    gradQ = YQ .*repmat(D2',p,1);

    ww = Q'*gradQ;
    G.Q = -gradQ + 0.5*Q*(ww+ww');
    G.D = -diag(Q'*YQ) .* D;
    G.Psi = -diag(Y).*Psi;
    

end


