function gbm = f3gbm_setup(numin, numout, nummap, numfactors, varargin)
% F3GBMSETUP
%
%
%

assert(isnumeric(numin));
assert(isnumeric(numout));
assert(isnumeric(nummap));
assert(isnumeric(numfactors));

gbm = f3gbm_init();
gbm = parseArgs([varargin] ,gbm);
gbm.numin = numin;
gbm.numout = numout;
gbm.nummap = nummap;
gbm.numfactors = numfactors;
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
end


function gbm = f3gbm_init()
% returns default value for arguments.
gbm = struct('numin',[],'numout',[],'nummap',256,'numfactors',1024, ...
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