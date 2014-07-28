function gbm = f4gbm_test(gbm, test_x, test_y, test_z, varargin)
% F4GBM_TEST
%
%
%
%

assert(size(test_x,1)==size(test_y,1));
assert(size(test_x,1)==size(test_z,1));
assert(size(test_x,2)==gbm.n_x);
assert(size(test_y,2)==gbm.n_y);
assert(size(test_z,2)==gbm.n_z);

mode = test_mode_init();
mode = parseArgs([varargin], mode);
assert(isempty(mode.NumericArguments));
mode = rmfield(mode,'NumericArguments');

testsize = size(test_x, 1);

X = test_x;
Y = zeros(size(test_y));
Z = test_z;
H = zeros(testsize, gbm.n_h);

% initialization
if mode.random_init == true
    H = double(rand(size(H))>0.5);
else
    if isequal(gbm.visType, 'binary')     
    else
        assert(isequal(gbm.visType, 'gaussian'));
        for index=1:gbm.n_h
            mu = (X*gbm.wxf).*(Z*gbm.wzf).*gbm.whf(index,:)*(gbm.wyf');
            H(index,:) = sigm(gbm.wh(index)+mu*(mu+2*gbm.wy));
        end
    end
end

for step = 1:mode.gibbs_steps
    H = double(H > rand(size(H)));
    if gbm.meanfield_output
        if isequal(gbm.visType, 'binary')
            Y = double(sigm((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(testsize,1)*gbm.wy') > rand(size(Y)));
        else
            assert(isequal(gbm.visType, 'gaussian'));
            Y = (X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(testsize,1)*gbm.wy' + randn(size(Y));
        end
    else
        if isequal(gbm.visType, 'binary')
            Y = sigm((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(testsize,1)*gbm.wy');
        else
            assert(isequal(gbm.visType, 'gaussian'));
            Y = (X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(testsize,1)*gbm.wy';
        end
    end
    H = sigm((X*gbm.wxf).*(Z*gbm.wzf).*(Y*gbm.wyf)*(gbm.whf') + ones(testsize,1)*gbm.wh');
end

if mode.show == true
    len = sqrt(gbm.n_x);
    for index = 1:testsize
        figure,
        subplot(2,3,1), imshow(imresize(reshape(test_x(index,:), [len, len]),20)), title('input x frame');
        subplot(2,3,2), imshow(imresize(reshape(Y(index,:), [len, len]),20)), title('predicted y frame');
        subplot(2,3,3), imshow(imresize(reshape(test_z(index,:), [len, len]),20)), title('input z frame');
        subplot(2,3,5), imshow(imresize(reshape(test_y(index,:), [len, len]),20)), title('groundtruth y frame');
        pause;
        close all;
    end
end

end


function mode = test_mode_init()
% returns default value for arguments.
mode = struct('gibbs_steps',10,'show',true,'random_init',false)
% seed is random seed. (twister).
end