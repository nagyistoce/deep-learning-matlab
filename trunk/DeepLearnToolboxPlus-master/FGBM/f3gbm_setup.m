function gbm = f3gbm_setup(n_x, n_y, n_h, n_f, varargin)
% F3GBMSETUP
%
%
%

assert(isnumeric(n_x));
assert(isnumeric(n_y));
assert(isnumeric(n_h));
assert(isnumeric(n_f));

gbm = f3gbm_init();
gbm = parseArgs([varargin] ,gbm);
gbm.n_x = n_x;
gbm.n_y = n_y;
gbm.n_h = n_h;
gbm.n_f = n_f;
assert(isempty(gbm.NumericArguments));
gbm = rmfield(gbm,'NumericArguments');

datestring = datestr(now,30);
gbm.datestring = datestring;

% intialize seed.
if isempty(gbm.seed)
    rng('shuffle');
else
    rng(gbm.seed,'twister');
end

% initialize weights.
if isempty(gbm.wxf) % in case we can override this.
    gbm.wxf = gbm.initMultiplierW*randn(gbm.n_x,gbm.n_f);
end

if isempty(gbm.wyf)
    gbm.wyf = gbm.initMultiplierW*randn(gbm.n_y,gbm.n_f);
end

if isempty(gbm.whf)
    gbm.whf = gbm.initMultiplierW*randn(gbm.n_h,gbm.n_f);
end

if isempty(gbm.wy)
    gbm.wy = zeros(gbm.n_y,1);
end

if isempty(gbm.wh)
    gbm.wh = zeros(gbm.n_h,1);
end

% initialize zeromask

gbm.zeromaskOriginal = gbm.zeromask;

if isequal(gbm.zeromask,'none')
    gbm.zeromask = false((gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f+gbm.n_y+gbm.n_h, 1);
elseif isequal(gbm.zeromask,'quadrature') % every hidden unit is connected to exactly 2 factors.
    % maybe extended to other things.
    assert(gbm.n_h*2 == gbm.n_f);
    
    zeroIndex = zeros(gbm.n_h*(gbm.n_f-2),1);
    
    for iMap = 1:gbm.n_h
        zeroIndex((iMap-1)*(gbm.n_f-2) + 1: (iMap)*(gbm.n_f-2))...
            = sub2ind(size(gbm.whf),repmat(iMap,1,gbm.n_f-2),setdiff(1:gbm.n_f,[2*iMap-1, 2*iMap]));
    end
       
    
    gbm.whf(zeroIndex) = 0; % test passed.
    
    if gbm.zeromaskAdditional
        gbm.whf(gbm.whf~=0) = 1; % all other weights to be one.
    end
    
    gbm.zeromask = false(size(gbm.inc));
    
    if ~gbm.zeromaskAdditional
        gbm.zeromask(zeroIndex+(gbm.n_x+gbm.n_y)*gbm.n_f) = true;
    else
        gbm.zeromask((gbm.n_x+gbm.n_y)*gbm.n_f+1:(gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f) = true;
    end
end

display(gbm);
end


function gbm = f3gbm_init()
% returns default value for arguments.
gbm = struct('n_x',[],'n_y',[],'n_h',256,'n_f',1024, ...
    'sgbmitygain',0.0, ...
    'targethidprobs', 0.1, 'cditerations', 1, ...
    'meanfield_output', false, 'momentum', 0.9, ...
    'stepsize', 0.01, 'verbose', true, 'zeromask', 'none',...
    'batchsize',500,'n_epoch',100,'seed',[],'initMultiplierW',0.05,...
    'batchOrderFixed',false,'weightPenaltyL2',0.001,'weightPenaltyL1',0.001,...
    'everySave',1,'wxf',[],'wyf',[],'whf',[],'wy',[],'wh',[],...
    'deltaMax',inf,'n_batch',[],'visType','binary','saveFile',true,'zeromaskOriginal',[],...
    'zeromaskAdditional','false');
% seed is random seed. (twister).
end