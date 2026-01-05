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

    p = size(corrmat,1); % number of variables

    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end

    if ~isfield(options, 'common_rank')
        options.common_rank = p-1;
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

    r = options.common_rank;
    condn = size(corrmat,3); % number of conditions

    % set random start point
    % [Q,Rw] = qr(rand(p,r)-.5,0); % random start point for Q
    % idx = find(diag(Rw)<0); Q(:,idx) = -Q(:,idx); % ensure positive diagonal in Rw
    % S.Q = Q; S.D = rand(r,condn) - .5; 

    % for i = 1:condn
    %     S.Psi(:,i) = diag(corrmat(:,:,i)-Q*diag(S.D(:,i).^2)*Q'); % initial parameters
    % end

    % create the problem structure
    elements = struct();
    elements.Q = stiefelfactory(p, r);
    elements.D = euclideanfactory(r,condn);
    elements.Psi = euclideanfactory(p,condn);
    M = productmanifold(elements);
    problem.M = M;

    % define the cost function and its gradient
    problem.costgrad = @(S) costgrad(corrmat, S.Q, S.D, S.Psi, options.sample_n, options.method);

    % Numerically check gradient consistency (just once, optional).
    % checkgradient(problem); pause;

    % use the eigenvector of the mean covariance matrix as initial point

    if ~isfield(options, 'init_point')
        [v_avg,d_avg] = eig(mean(corrmat,3));
        [d_avg,idx] = sort(diag(d_avg),'descend');
        v_avg = v_avg(:,idx);
        options.init_point.Q = v_avg(:,1:r);
        options.init_point.D = repmat(sqrt(d_avg(1:r)),1,condn);
        for i = 1:condn
            options.init_point.Psi(:,i) = diag(corrmat(:,:,i)-options.init_point.Q*diag(options.init_point.D(:,i).^2)*options.init_point.Q'); % initial parameters
        end
    end

    % Now solve 
    [result, cost, info] = trustregions(problem, options.init_point, options.manopt);
    
    % the eigenvalues and the psi should be non-negative
    result.D = abs(result.D);
    result.Psi = abs(result.Psi);

    % sort the common factors by the average eigenvalues
    [~,idx] = sort(mean(result.D.^2,2),'descend');
    result.Q = result.Q(:,idx);
    result.D = result.D(idx,:);

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

function [f, G] = costgrad(R,Q,D,Psi,sample_n,method)
    [p,r] = size(Q);
    D2 = D.^2;
    Psi2 = Psi.^2;

    RM = zeros(p,p,size(R,3));
    for cond = 1:size(R,3)
        QD = Q .* repmat(D(:,cond)',p,1);
        RM(:,:,cond) = QD * QD' + diag(Psi2(:,cond));
    end

    Y = R - RM; % residuals

    % objective function
    switch method
        case 'ML'
            f = zeros(size(R,3),1);
            for cond = 1:size(R,3)
                Y(:,:,cond) = (RM(:,:,cond)\Y(:,:,cond))/RM(:,:,cond);
                f(cond) = sample_n(cond) .* (log(det(RM(:,:,cond))) + trace(RM(:,:,cond)\R(:,:,cond)))/2;
            end
            f = sum(f);
        case 'GLS'
            f = zeros(size(R,3),1);
            for cond = 1:size(R,3)
                W = Y(:,:,cond)/R(:,:,cond);
                Y(:,:,cond) = R(:,:,cond)\W;
                f(cond) = sample_n(cond) .* trace(W*W)/4;
            end
            f = sum(f);
        case 'LS'
            f = zeros(size(R,3),1);
            for cond = 1:size(R,3)
                f(cond) = sample_n(cond) .* trace(Y(:,:,cond)'*Y(:,:,cond))/4;
            end
            f = sum(f);
        otherwise
            error('Unknown method');
    end


    % gradient
    G.Q = zeros(p,r,size(R,3));
    for cond = 1:size(R,3)
        YQ = Y(:,:,cond) * Q;
        gradQ = YQ .* repmat(D2(:,cond)',p,1);

        ww = Q'*gradQ;
        G.Q(:,:,cond) = sample_n(cond) .* (-gradQ + 0.5*Q*(ww+ww'));
    end
    G.Q = sum(G.Q,3); % sum over conditions

    for cond = 1:size(R,3)
        G.D(:,cond) = -diag(Q'*Y(:,:,cond)*Q) .* D(:,cond) .* sample_n(cond);
        G.Psi(:,cond) = -diag(Y(:,:,cond)) .* Psi(:,cond) .* sample_n(cond);
    end
    

end


