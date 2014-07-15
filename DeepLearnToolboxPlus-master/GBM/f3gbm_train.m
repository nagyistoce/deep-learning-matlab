function gbm = f3gbm_train(gbm, train_x, train_y)
% F3GBMTRAIN
%
%
%
%

assert(size(train_x,1)==size(train_y,1));
assert(size(train_x,2)==gbm.numin);
assert(size(train_y,2)==gbm.numout);

numbatches = floor(size(train_x,1)/gbm.batchsize);

% training
for epoch = 1:gbm.numepoch
    fprintf('epoch %d\n',epoch);
    if ~gbm.batchOrderFixed
        order = randperm(size(train_x,1));
    else
        order = 1:size(train_x,1);
    end
    
    for batch = 1:numbatches
        batch_x = train_x(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        batch_y = train_y(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        f3gbm_train_inner(); 
    end
    
    if rem(epoch, gbm.everySave) == 0 || epoch == gbm.numepoch
        if gbm.saveFile
            fileName = [gbm.datestring '_' int2str(epoch) '.mat'];
            save(fileName,'gbm','epoch'); 
        end
    end  
end

    function f3gbm_train_inner()
        gbm.inc = gbm.momentum*gbm.inc - gbm.stepsize*f3gbm_grad();
        
        % zero mask used here
        gbm.inc(gbm.zeromask) = 0;
        
        fprintf('epoch %d ## norm of inc: %f\n', epoch ,norm(gbm.inc));
        
        % stablize the things...
        if norm(gbm.inc) > gbm.incMax
            gbm.inc = gbm.inc/ninc * gbm.incMax;
            fprintf('epoch %d ## norm of inc again: %f\n', epoch ,norm(gbm.inc));
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
            fprintf('epoch %d ## mean square error: %f\n', epoch , sum( (Y(:)-batch_y(:)).^2) / gbm.batchsize   );
            fprintf('epoch %d ## mean norm of w: %f\n', epoch , norm([gbm.wxf(:); gbm.wyf(:); gbm.whf(:); gbm.wh(:); gbm.wy(:)]));
        end
             
        
        %%
        save positiveGrad positiveGrad
        save negativeGrad negativeGrad
        %%
        
        grad = -positiveGrad + negativeGrad;
        
        weightcostgrad_x = gbm.weightPenaltyL2 * gbm.wxf(:);
        weightcostgrad_y = gbm.weightPenaltyL2 * gbm.wyf(:);
        weightcostgrad_h = gbm.weightPenaltyL2 * gbm.whf(:);
        
        weightcostgrad = [weightcostgrad_x; weightcostgrad_y; weightcostgrad_h];
        
        grad(1:(gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors) = ...
            grad(1:(gbm.numin+gbm.numout+gbm.nummap)*gbm.numfactors) + weightcostgrad;
    end

end

