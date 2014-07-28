function gbm = f4gbm_train(gbm, train_x, train_y, train_z)
% F3GBMTRAIN
%
%
%
%

assert(size(train_x,1)==size(train_y,1));
assert(size(train_x,1)==size(train_z,1));
assert(size(train_x,2)==gbm.n_x);
assert(size(train_y,2)==gbm.n_y);
assert(size(train_z,2)==gbm.n_z);

n_batch = floor(size(train_x,1)/gbm.batchsize);
% delta for momentum
delta = zeros((gbm.n_x+gbm.n_y+gbm.n_z+gbm.n_h)*gbm.n_f+gbm.n_y+gbm.n_h,1);
d_pos = [0, gbm.n_x*gbm.n_f, (gbm.n_x+gbm.n_y)*gbm.n_f, (gbm.n_x+gbm.n_y+gbm.n_z)*gbm.n_f, ...
    (gbm.n_x+gbm.n_y+gbm.n_z+gbm.n_h)*gbm.n_f, (gbm.n_x+gbm.n_y+gbm.n_z+gbm.n_h)*gbm.n_f+gbm.n_y,...
    (gbm.n_x+gbm.n_y+gbm.n_z+gbm.n_h)*gbm.n_f+gbm.n_y+gbm.n_h];

% training
for epoch = 1:gbm.n_epoch
    if ~gbm.batchOrderFixed
        order = randperm(size(train_x,1));
    else
        order = 1:size(train_x,1);
    end
    
    for batch = 1:n_batch
        batch_x = train_x(order((batch-1)*gbm.batchsize+1:batch*gbm.batchsize),:);
        batch_y = train_y(order((batch-1)*gbm.batchsize+1:batch*gbm.batchsize),:);
        batch_z = train_z(order((batch-1)*gbm.batchsize+1:batch*gbm.batchsize),:);
        f3gbm_train_inner();
    end
    
    if rem(epoch, gbm.everySave) == 0 || epoch == gbm.n_epoch
        if gbm.saveFile
            fileName = [gbm.datestring '_' int2str(epoch) '.mat'];
            save(fileName,'gbm','epoch');
        end
    end
end

    function f3gbm_train_inner()
        delta = gbm.momentum*delta - gbm.stepsize*f3gbm_grad();
        
        % zero mask used here
        delta(gbm.zeromask) = 0;
        
        % stablize the things
        if norm(delta) > gbm.deltaMax
            delta = delta/ninc * gbm.deltaMax;
        end
        
        assert(all(~isnan(delta)));
        
        W = [gbm.wxf(:);gbm.wyf(:);gbm.wzf(:);gbm.whf(:);gbm.wy(:);gbm.wh(:)];
        
        % regularization
        if gbm.weightPenaltyL2 > 0
            W = W - gbm.weightPenaltyL2*W;
        end
        if gbm.weightPenaltyL1 > 0
            W = W - (abs(W+gbm.weightPenaltyL1)-abs(W-gbm.weightPenaltyL1))/2;
        end
        
        W = W + delta;
        
        gbm.wxf(:) = W(d_pos(1)+1:d_pos(2));
        gbm.wyf(:) = W(d_pos(2)+1:d_pos(3));
        gbm.wzf(:) = W(d_pos(3)+1:d_pos(4));
        gbm.whf(:) = W(d_pos(4)+1:d_pos(5));
        gbm.wy(:)  = W(d_pos(5)+1:d_pos(6));
        gbm.wh(:)  = W(d_pos(6)+1:d_pos(7));
        
    end

    function grad = f3gbm_grad()
        X = batch_x;
        Y = batch_y;
        Z = batch_z;
        H = sigm((X*gbm.wxf).*(Y*gbm.wyf).*(Z*gbm.wzf)*(gbm.whf') + ones(gbm.batchsize,1)*gbm.wh');
        
        g_wxf = -X'*((Y*gbm.wyf).*(Z*gbm.wzf).*(H*gbm.whf))/gbm.batchsize;
        g_wyf = -Y'*((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf))/gbm.batchsize;
        g_wzf = -Z'*((X*gbm.wxf).*(Y*gbm.wyf).*(H*gbm.whf))/gbm.batchsize;
        g_whf = -H'*((X*gbm.wxf).*(Y*gbm.wyf).*(Z*gbm.wzf))/gbm.batchsize;
        g_wy  = -mean(Y,1)';
        g_wh  = -mean(H,1)';
        
        for iCD = 1:gbm.cditerations
            H = double(H > rand(size(H)));
            if ~gbm.meanfield_output
                if isequal(gbm.visType, 'binary')
                    Y = double(sigm((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy') > rand(size(Y)));
                else
                    assert(isequal(gbm.visType, 'gaussian'));
                    Y = (X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy' + randn(size(Y));
                end
            else
                if isequal(gbm.visType, 'binary')
                    Y = sigm((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy');
                else
                    assert(isequal(gbm.visType, 'gaussian'));
                    Y = (X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf)*(gbm.wyf') + ones(gbm.batchsize,1)*gbm.wy';
                end
            end
            H = sigm((X*gbm.wxf).*(Z*gbm.wzf).*(Y*gbm.wyf)*(gbm.whf') + ones(gbm.batchsize,1)*gbm.wh');
        end
        
        g_wxf = g_wxf + X'*((Y*gbm.wyf).*(Z*gbm.wzf).*(H*gbm.whf))/gbm.batchsize;
        g_wyf = g_wyf + Y'*((X*gbm.wxf).*(Z*gbm.wzf).*(H*gbm.whf))/gbm.batchsize;
        g_wzf = g_wzf + Z'*((X*gbm.wxf).*(Y*gbm.wyf).*(H*gbm.whf))/gbm.batchsize;
        g_whf = g_whf + H'*((X*gbm.wxf).*(Y*gbm.wyf).*(Z*gbm.wzf))/gbm.batchsize;
        g_wy  = g_wy  + mean(Y,1)';
        g_wh  = g_wh  + mean(H,1)';
        
        if gbm.verbose % print reconstruction error and the norm of weights
            Y = double(Y > 0.5);
            fprintf('#epoch %d# sum square error: %f , norm of w: %f\n', epoch, sum((Y(:)-batch_y(:)).^2)/gbm.batchsize,...
                norm([gbm.wxf(:); gbm.wyf(:); gbm.wzf(:); gbm.whf(:); gbm.wh(:); gbm.wy(:)]));
        end
        
        grad = [g_wxf(:);g_wyf(:);g_wzf(:);g_whf(:);g_wy(:);g_wh(:)];
    end

end

