function rbm = rbmtrain(rbm, x, hintonFlag, saveFlag, fileName)
assert(isfloat(x), 'x must be a float');
% assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');

if nargin < 4
    saveFlag = false;
end

m = size(x, 1);

numbatches = m / rbm.batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% can be commented out

if nargin < 3
    hintonFlag = false;
end

if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
    if isfield(rbm,'sparsityTargetStd') && rbm.sparsityTargetStd > 0 % make a vector of sparsity target
	rbm.sparsityTargetOld = rbm.sparsityTarget;
        rbm.sparsityTarget = rbm.sparsityTargetOld + rbm.sparsityTargetStd*randn(1,size(rbm.W,1));
        
        fprintf('%d units with negative sparsity target\n',sum(rbm.sparsityTarget<=0));

        rbm.sparsityTarget(rbm.sparsityTarget<=0) = rbm.sparsityTargetOld ...
            - rbm.sparsityTargetStd; % make those outliers to be one sigma smaller than the mean.
        
        assert(sum(rbm.sparsityTarget<=0)==0);
        
        fprintf('mean target %f, std target %f\n',mean(rbm.sparsityTarget), std(rbm.sparsityTarget));
        
    end
end



display(rbm);

initMultiplierW = rbm.initMultiplierW;

if ~isfield(rbm,'initialized') || ~rbm.initialized
    % add an initialization phase.
    rbm.W = initMultiplierW*randn(size(rbm.W,2),size(rbm.W,1))'; % follow the initialization of hinton
    % 0.1 can be changed. In Ruslan's DBM code, this value can be sometimes
    % 0.01, sometimes 0.001.
end

% lateral connections for visible
% diagonals are zero
% L_{ij} = L_{ji}, and they represent the connection between vi and vj.
% when computing the whole energy, should use 0.5 * V^T * L * V, since
% double counting.
% I store them this way mainly for mean-field update using matrix notation.

% there should be some mask to indicate what connections are needed, and
% what are not.
% to implement this correctly, I'd better write two subfunctions, one naive
% MF (no matrix notation), and one normal version

if isfield(rbm,'lateralVisible') && rbm.lateralVisible
    if any(rbm.lateralVisibleMask(:) == true)
        initMultiplierLV = rbm.initMultiplierLV;
        
        rbm.LV = initMultiplierLV*randn(size(rbm.W,2),size(rbm.W,2));  % LV means laterval visible. % how big should they are ...
    end
    rbm.LV = (rbm.LV + rbm.LV')/2; % must be symmetric.
    LVdiagIndex = logical(eye(size(rbm.LV)));
    rbm.LV(LVdiagIndex) = 0;
    
    % this part is done in dbnsetup.
    %     if ~isfield(rbm,'lateralVisibleMask') % if there's a 1, it indicates connection.
    %
    %
    %         rbm.lateralVisibleMask = true(size(rbm.LV));
    %
    % %         if hintonFlag
    % %             rbm.lateralVisibleMask = false(size(rbm.LV)); % hack to modify connectivity.
    % %         end
    %
    %         rbm.lateralVisibleMask(LVdiagIndex) = false;
    %     end
    
    % rbm.lateralVisibleMask is a symmetric matrix, with diagonals 0.
    
    assert(isequal(rbm.lateralVisibleMask,rbm.lateralVisibleMask'));
    assert(all(diag(rbm.lateralVisibleMask)==false));
    
    rbm.LV(~rbm.lateralVisibleMask) = 0; % wherever there's false, we set that entry in LV to zero.
    % note that diagonals must be false.
    
    assert(all(   diag(rbm.LV)==0    ));
    assert(all(   rbm.LV(~rbm.lateralVisibleMask)==0  ));
end

if hintonFlag && rbm.lateralVisible && all(rbm.lateralVisibleMask(:) == false)
    rbm.xlast = cell(numbatches,1);
end

gaussianLayerCount = 0;
for i = 1:2
    assert(isequal(rbm.types{i},'binary') || ...
        isequal(rbm.types{i},'gaussian'));
    if isequal(rbm.types{i},'gaussian')
        gaussianLayerCount = gaussianLayerCount+1;
    end
end
assert(gaussianLayerCount<=1);

sigma = rbm.sigma;

for i = 1 : rbm.numepochs
    tic;
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        sparsity = 0;
    end
    
    if ~rbm.batchOrderFixed
        kk = randperm(m);
    else
        kk = 1:m;
    end
    err = 0;
    
    for l = 1 : numbatches
        batch = x(kk((l - 1) * rbm.batchsize + 1 : l * rbm.batchsize), :);
        
        v1 = batch; % this is N x D.
        
        if isequal(rbm.types{1},'binary')
            h1 = sigm(  (1/(sigma^2))  *  (v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1) )     );
            %             h1 = sigmrnd(v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1));
        else
            h1 = v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
        end
        
        if hintonFlag && rbm.lateralVisible && all(rbm.lateralVisibleMask(:) == false) % save last states, to match hinton
            rbm.xlast{l} = h1;
        end
        
        c1 = (v1'*h1)'; % positive statistics.
        
        %             c1_alter = h1'*v1;
        %             assert(isequal(c1_alter,c1));
        
        poshidact   = sum(h1);
        posvisact = sum(batch);
        
        
        if isfield(rbm,'lateralVisible') && rbm.lateralVisible
            posvisvis = (v1')*v1;
            % this thing should not be halved later, since each entry in LV
            % is the full weight, not halved version.
            % should write something naive to check my above statement.
        end
        
        
        h2 = h1;
        
        for iCDIter = 1:rbm.CDIter % do CDIter-CD.
            % h2 is the hidden layer probability from last step of CD, or
            % the probability from positive phase (for first step in CD).
            % following Hinton's recommendation, we should always
            % reconstruct based on a 0-1 sampled hidden layer, but when
            % infer hidden from visible, the visible can be real-valued.
            
            
            if isequal(rbm.types{1},'binary') %stochastic sample based on h2.
                h2Sampled = h2 > rand(size(h1));
                %             h1Sampled = h1;
            else
                h2Sampled = h2 + sigma*randn(size(h1)); % sigma is the std deviation
                % why my current implementation fail to match old one on
                % vanHateren_Lee
            end
            
            
            % reconstruction
            if isequal(rbm.types{2},'binary')
                v2 = sigm(  (1/(sigma^2))*    (h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1))         );
                %             v2 = sigmrnd(h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1));
            else
                v2 = h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1);
                % let's try sampled version...
                %             v2 = h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1)...
                %                  + (sigma^2)*randn(rbm.batchsize, size(rbm.W,2)) ; ZYM:
                %                  why my current result on vanHateren_Lee failed to match
                %                  old one.
            end
            
            
            % the input v2 is the initial mean of v2.
            if isfield(rbm,'lateralVisible') && rbm.lateralVisible
                % update v2 using mean-field so that they somewhat converge
                [v2] = rbm_meanfield(v2, h2Sampled, rbm, sigma);
                %                 disp(MFiter);
            end
            
            
            if isequal(rbm.types{1},'binary') % h2 is probability, and h2Sampled is stochastic version.
                h2 = sigm(   (1/(sigma^2)) *(v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1))     );
            else
                h2 = v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
            end
            
        end
        
        c2 = (v2'*h2)'; % this can affect result?!!! order of computation...
        
        %             c2_alter = h2'*v2;
        %             assert(isequal(c2_alter,c2)); this can indeed affect
        %             computation...
        
        
        if isfield(rbm,'lateralVisible') && rbm.lateralVisible
            negvisvis = (v2')*v2;
            % this thing should not be halved later, since each entry in LV
            % is the full weight, not halved version.
            % should write something naive to check my above statement.
        end
        
        
        if i > rbm.epochFinal
            momentum=rbm.momentumFinal;
            alpha = rbm.alphaFinal;
        else
            momentum=rbm.momentum;
            alpha = rbm.alpha;
        end
        
        neghidact = sum(h2);
        negvisact = sum(v2);
        if mod(l,50) == 0
            disp(['dW norm: ' num2str( mean((rbm.vW(:)).^2) ) ]);
        end
        % for vW, I put division afterwards, to make my program behave
        % exactly as Ruslan's autoencoder.
        rbm.vW = momentum * rbm.vW + alpha * ((c1 - c2)/rbm.batchsize - rbm.weightPenaltyL2*rbm.W);
        rbm.vb = momentum * rbm.vb + alpha/rbm.batchsize * ((posvisact-negvisact)');
        % put division first, so that we may have better accuracy?
        rbm.vc = momentum * rbm.vc + alpha/rbm.batchsize * ((poshidact-neghidact)');
        
        if isfield(rbm,'lateralVisible') && rbm.lateralVisible  % weight decay to LV.
            rbm.vLV = momentum * rbm.vLV + rbm.alphaLateral * ( (posvisvis-negvisvis)/rbm.batchsize  -rbm.weightPenaltyL2*rbm.LV);
            rbm.vLV(~rbm.lateralVisibleMask) = 0; % remove those unrelated term
            assert(max(max(abs(rbm.vLV-rbm.vLV')))==0); % should be zero.
            % now, this alphaLateral has no 'final' version...
            
            if mod(l,50) == 0
                disp(['dLV norm: ' num2str( mean((rbm.vLV(:)).^2) ) ]);
            end
            
        end
        
        
        
        assert(all(~isnan(rbm.vW(:))));
        assert(all(~isnan(rbm.vb(:))));
        assert(all(~isnan(rbm.vc(:))));
        
        rbm.W = rbm.W + rbm.vW;
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;
        
        if isfield(rbm,'lateralVisible') && rbm.lateralVisible
            rbm.LV = rbm.LV + rbm.vLV;
            rbm.LV(~rbm.lateralVisibleMask) = 0;
        end
        
        err = err + sum(sum((v1 - v2) .^ 2)) / rbm.batchsize;
        
    end
    
    if i > rbm.epochFinal
        fprintf('finally!, with alpha %f, momentum %f\n', alpha, momentum);
    else
        fprintf('initial!, with alpha %f, momentum %f\n', alpha, momentum);
    end
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        assert(isfield(rbm,'sparsityTarget'));
        assert(isequal(rbm.types{1},'binary')); % only works for binary hidden layer
        
        if ~isscalar(rbm.sparsityTarget)
            fprintf('vectorized sparsity!');
        end
        
        v1All = x;
        
        if isequal(rbm.types{1},'binary')
            h1All = sigm( (1/(sigma^2))  *  (v1All * rbm.W' + repmat(rbm.c', m, 1) )     );
        else
            h1All = v1All * rbm.W' + repmat(rbm.c', m, 1);
        end
        % h1All is a matrix of size [N x hiddenSize].
        
        %         sparsityGradientSecondTerm = sum(h1All.*(1-h1All),1);
        sparsityGradientFirstTerm =  rbm.sparsityTarget - mean(h1All,1);
        % sparsityGradientFirstTerm is p-q in Hinton's TR. also the
        % negative gradient for the bias.
        % sparsityGradientFirstTerm is of size [1 x hiddenSize].
        
%         sparsityGradientW = (rbm.sparsityTarget - h1All'); % [hiddenSize x N]
        sparsityGradientW = bsxfun(@minus, rbm.sparsityTarget, h1All); %[N x hiddenSize]
        sparsityGradientW = sparsityGradientW';
        sparsityGradientW = (1/m) * (sparsityGradientW*v1All); % [hidden x visible] 1/m for average.
        
        sparsity = mean(h1All(:));
        if isnan(sparsity)
            error('what the fuck!\n');
        end
    end
    
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        %         rbm.c = rbm.c + (rbm.nonSparsityPenalty/m) *...
        %             (sparsityGradientFirstTerm.*sparsityGradientSecondTerm)';
        fprintf('simple form!\n');
        rbm.c = rbm.c + (rbm.nonSparsityPenalty) *... %no 1/m
            (sparsityGradientFirstTerm)';
        
        if isfield(rbm,'nonSparsityPenaltyOnW') && rbm.nonSparsityPenaltyOnW
            % if we like Hinton's recommendation
            rbm.W = rbm.W + (rbm.nonSparsityPenalty) * sparsityGradientW;
            fprintf('update W for sparsity!\n');
        end
        
    end
    
    fprintf('sigma %f\n',sigma);
    
    rbm.sigmaFinal = sigma; % save the current sigma...
    
    if sigma > rbm.sigmaMin % sigma decay in Honglak
        sigma = sigma*rbm.sigmaDecay; %sigmaDecay is 1 by default.
    end
    toc;
    disp(['epoch ' num2str(i) '/' num2str(rbm.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    
    disp(['W norm: ' num2str( mean((rbm.W(:)).^2) ) ]);
    
    
    if isfield(rbm,'lateralVisible') && rbm.lateralVisible  % weight decay to LV.
        
        if mod(l,50) == 0
            disp(['LV norm: ' num2str( mean((rbm.LV(:)).^2) ) ]);
        end
        
    end
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        fprintf('sparsity is %f\n', sparsity);
    end
    
    if isfield(rbm,'visualize') && rbm.visualize
        visualize(rbm.W');
        title(num2str(i));
        drawnow;
        print('-depsc2',[fileName '.eps']);
    end
    
    if saveFlag
        errAvg = err / numbatches;
        
        if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
            save(fileName,'rbm','sparsity','i','errAvg');
        else
            save(fileName,'rbm','i','errAvg');
        end
    end
    
end



end
