function gbm = f3gbmsetup(n_x, n_y, n_h, n_f, varargin)
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
gbm.Wxf = gbm.initMultiplierW*randn(n_x,n_f);
gbm.Wyf = gbm.initMultiplierW*randn(n_y,n_f);
gbm.Whf = gbm.initMultiplierW*randn(n_h,n_f);
gbm.wx  = zeros(n_x,1);
gbm.wy  = zeros(n_y,1);
gbm.wh  = zeros(n_h,1);

% initialize zeromask
% (TO BE CONTINUED)

display(gbm);

    function xx = xx()
        gbm = gbm();
    end
end


function gbm = f3gbm_init()
% returns default value for arguments.
gbm = struct('sparsitygain',0.0, ...
    'targethidprobs', 0.1, 'cditerations', 1, ...
    'meanfield_output', false, 'momentum', 0.9, ...
    'stepsize', 0.01, 'verbose', true, 'zeromask', 'none',...
    'batchsize',500,'n_epoch',100,'seed',[],'initMultiplierW',0.05,...
    'batchOrderFixed',false,'weightPenaltyL2',0.001,'epochfinished',0,...
    'everySave',1,'deltaMax',inf,'n_batches',[],'visType','binary',...
    'saveFile',true,'zeromaskOriginal',[],'zeromaskAdditional','false');
% seed is random seed. (twister).
end