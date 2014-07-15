function gbm = f3gbm_train(varargin)

train_x = varargin{1};
train_y = varargin{2};

assert(ismatrix(train_x));
assert(ismatrix(train_y));

[N,numin] = size(train_x);
numout = size(train_y,2);

assert(N == size(train_y,1));

gbm = factor_GBM_default_gbm();
% fill in other values, along with numin and numout, inferred from data.
gbm = parseArgs([varargin(3:end),{'numin'},{numin},{'numout'},{numout} ]  ,gbm);
assert(isempty(gbm.NumericArguments));
gbm = rmfield(gbm,'NumericArguments');

datestring = datestr(now,30);
gbm.datestring = datestring;

assert(rem(N,gbm.batchsize)==0);
numbatches = N/gbm.batchsize;
everySave = gbm.everySave;

% intialize seed.
if isempty(gbm.seed)
    rng('shuffle');
else
    rng(gbm.seed,'twister');
end


if ~isempty(gbm.numbatches) % hack for early quitting.
    numbatches = gbm.numbatches;
end

gbm.numbatches = numbatches;

% initialize weights.
if isempty(gbm.wxf) % in case we can override this.
    gbm.wxf = gbm.initMultiplierW*randn(gbm.numin,gbm.numfactors);
end

if isempty(gbm.wyf)
    gbm.wyf = gbm.initMultiplierW*randn(gbm.numout,gbm.numfactors);
end

if isempty(gbm.whf)
    gbm.whf = gbm.initMultiplierW*randn(gbm.nummap,gbm.numfactors);
end

if isempty(gbm.wy)
    gbm.wy = zeros(gbm.numout,1);
end

if isempty(gbm.wh)
    gbm.wh = zeros(gbm.nummap,1);
end

gbm.inc = zeros(  (gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors + gbm.numout + gbm.nummap,1 );

% initialize zeromask

gbm.zeromaskOriginal = gbm.zeromask;

if isequal(gbm.zeromask,'none')
    gbm.zeromask = false(size(gbm.inc));
elseif isequal(gbm.zeromask,'quadrature') % every hidden unit is connected to exactly 2 factors.
    % maybe extended to other things.
    assert(gbm.nummap*2 == gbm.numfactors);
    
    zeroIndex = zeros(gbm.nummap*(gbm.numfactors-2),1);
    
    for iMap = 1:gbm.nummap
        zeroIndex((iMap-1)*(gbm.numfactors-2) + 1: (iMap)*(gbm.numfactors-2))...
            = sub2ind(size(gbm.whf),repmat(iMap,1,gbm.numfactors-2),setdiff(1:gbm.numfactors,[2*iMap-1, 2*iMap]));
    end
    
    
    
    
    
    gbm.whf(zeroIndex) = 0; % test passed.
    
    if gbm.zeromaskAdditional
        gbm.whf(gbm.whf~=0) = 1; % all other weights to be one.
    end
    
    gbm.zeromask = false(size(gbm.inc));
    
    if ~gbm.zeromaskAdditional
        gbm.zeromask(zeroIndex+(gbm.numin+gbm.numout)*gbm.numfactors) = true;
    else % gbm.whf is now fixed.
        gbm.zeromask((gbm.numin+gbm.numout)*gbm.numfactors+1:(gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors) = true;
    end
end


display(gbm);
pause;

for epoch = 1:gbm.numepoch
    fprintf('epoch %d\n',epoch);
    if ~gbm.batchOrderFixed
        order = randperm(N);
    else
        order = 1:N;
    end
    
    for batch = 1:numbatches
        batch_x = train_x(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        batch_y = train_y(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        f3gbm_train_inner(); % here, I use a nested function for efficieny issue.
    end
    
    if rem(epoch,everySave) == 0 || epoch == gbm.numepoch
        if gbm.saveFile
            fileName = [datestring '_' int2str(epoch) '.mat'];
            gbm = rmfield(gbm,'hids'); % save space...
            save(fileName,'gbm','epoch'); % remember delete gbm.hids before!
        end
    end
    
end

if isfield(gbm,'hids')
    gbm = rmfield(gbm,'hids'); % save space...
end

    function f3gbm_train_inner()
        gradThis = f3gbm_grad();
        gbm.inc = gbm.momentum*gbm.inc - gbm.stepsize*gradThis;
        
        % set something to zero.
        gbm.inc(gbm.zeromask) = 0;
        
        ninc =norm(gbm.inc);
        fprintf('norm of inc: %f\n',ninc);
        
        % stablize the things...
        if norm(gbm.inc) > gbm.incMax
            gbm.inc = gbm.inc/ninc * gbm.incMax;
            fprintf('norm of inc again: %f\n',norm(gbm.inc));
        end
        
        assert(all(~isnan(gbm.inc)));
        
        % here, I should add them separately.
        gbm.wxf(:) = gbm.wxf(:) + gbm.inc( 1:gbm.numin*gbm.numfactors );
        gbm.wyf(:) = gbm.wyf(:) + gbm.inc( gbm.numin*gbm.numfactors+1 : (gbm.numin+gbm.numout)*gbm.numfactors );
        gbm.whf(:) = gbm.whf(:) + gbm.inc( (gbm.numin+gbm.numout)*gbm.numfactors+1 : (gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors );
        
        gbm.wy(:) = gbm.wy(:) + gbm.inc( (gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors+1 : ...
            (gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors + gbm.numout);
        
        gbm.wh(:) = gbm.wh(:) + gbm.inc( (gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors + gbm.numout + 1 : end );
        
    end

    function grad = f3gbm_grad()
        X = batch_x;
        Y = batch_y;
        H = sigm((X*gbm.wxf).*(Y*gbm.wyf)*(gbm.whf') + ones(gbm.batchsize,1)*gbm.wh');
        
        g_wxf = X'*((Y*gbm.wyf).*(H*gbm.whf))/gbm.batchsize;
        g_wyf = Y'*((X*gbm.wxf).*(H*gbm.whf))/gbm.batchsize;
        g_whf = H'*((X*gbm.wxf).*(Y*gbm.wyf))/gbm.batchsize;
        g_wy  = mean(Y,1)';
        g_wh  = mean(H,1)';
        
        positiveGrad = [g_wxf(:);g_wyf(:);g_whf(:);g_wy(:);g_wh(:)];
        
        
        
        for iCD = 1:gbm.cditerations
            if ~gbm.meanfield_output
                H = double(H > rand(size(H)));
            end
            %negoutput = (X*gbm.wxf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy';
            if ~gbm.meanfield_output
                if isequal(gbm.visType, 'binary')
                    Y = double(sigm((X*gbm.wxf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy') > rand(size(Y)));
                else
                    assert(isequal(gbm.visType, 'gaussian'));
                    Y = (X*gbm.wxf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy' + randn(size(Y));
                end
            else
                if isequal(gbm.visType, 'binary')
                    Y = sigm((X*gbm.wxf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy');
                else
                    assert(isequal(gbm.visType, 'gaussian'));
                    Y = (X*gbm.wxf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy';
                end
            end
            H = sigm((X*gbm.wxf).*(Y*gbm.wyf)*(gbm.whf') + ones(gbm.batchsize,1)*gbm.wh');
        end
        
        g_wxf = X'*((Y*gbm.wyf).*(H*gbm.whf))/gbm.batchsize;
        g_wyf = Y'*((X*gbm.wxf).*(H*gbm.whf))/gbm.batchsize;
        g_whf = H'*((X*gbm.wxf).*(Y*gbm.wyf))/gbm.batchsize;
        g_wy  = mean(Y,1)';
        g_wh  = mean(H,1)';
        
        negativeGrad = [g_wxf(:);g_wyf(:);g_whf(:);g_wy(:);g_wh(:)];
        
        
        if gbm.verbose % output recon and norm
            fprintf('mean square error: %f\n', sum( (Y(:)-batch_y(:)).^2) / gbm.batchsize   );
            fprintf('mean norm of w: %f\n', norm([gbm.wxf(:); gbm.wyf(:); gbm.whf(:); gbm.wh(:); gbm.wy(:)]));
        end
        
        
        
        
        
        %%
        save positiveGrad positiveGrad
        save negativeGrad negativeGrad
        %%
        grad = -positiveGrad + negativeGrad;
        grad = grad + sgbmityGrad;
        
        weightcostgrad_x = gbm.weightPenaltyL2 * gbm.wxf(:);
        weightcostgrad_y = gbm.weightPenaltyL2 * gbm.wyf(:);
        weightcostgrad_h = gbm.weightPenaltyL2 * gbm.whf(:);
        
        weightcostgrad = [weightcostgrad_x; weightcostgrad_y; weightcostgrad_h];
        
        grad(1:(gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors) = ...
            grad(1:(gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors) + weightcostgrad;
    end

end

function defaultgbm = factor_GBM_default_gbm()
% returns default value for arguments.
defaultgbm = struct('numin',[],'numout',[],'nummap',256,'numfactors',1024, ...
    'sgbmitygain',0.0, ...
    'targethidprobs', 0.1, 'cditerations', 1, ...
    'meanfield_output', false, 'momentum', 0.9, ...
    'stepsize', 0.01, 'verbose', true, ...
    'zeromask', 'none','batchsize',500,'numepoch',100,'seed',[],...
    'initMultiplierW',0.05,'batchOrderFixed',false,'weightPenaltyL2',0.001,...
    'everySave',1,'wxf',[],'wyf',[],'whf',[],'wy',[],'wh',[],...
    'incMax',inf,'numbatches',[],'visType','binary','saveFile',true,'zeromaskOriginal',[],...
    'zeromaskAdditional','false');
% seed is random seed. (twister).
end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [factor_GBM_train.m] ======
