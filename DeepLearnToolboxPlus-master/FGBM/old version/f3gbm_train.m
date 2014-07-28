function gbm = f3gbm_train(gbm, train_x, train_y)
% F3GBMTRAIN
%
%
%
%

assert(size(train_x,1)==size(train_y,1));
assert(size(train_x,2)==gbm.n_x);
assert(size(train_y,2)==gbm.n_y);

n_batch = floor(size(train_x,1)/gbm.batchsize);
% delta for momentum
delta = zeros((gbm.n_x+gbm.n_y+gbm.n_h)*(gbm.n_f+1),1);
d_pos = [0, gbm.n_x*gbm.n_f, (gbm.n_x+gbm.n_y)*gbm.n_f, (gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f, ...
    (gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f+gbm.n_x, (gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f+gbm.n_x+gbm.n_y, ...
    (gbm.n_x+gbm.n_y+gbm.n_h)*gbm.n_f+gbm.n_x+gbm.n_y+gbm.n_h];

% training
for epoch = 1:gbm.n_epoch
    fprintf('epoch %d\n',epoch);
    if ~gbm.batchOrderFixed
        order = randperm(size(train_x,1));
    else
        order = 1:size(train_x,1);
    end
    
    for batch = 1:n_batch
        batch_X = train_x(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        batch_Y = train_y(order((batch - 1) * gbm.batchsize + 1 : batch * gbm.batchsize), :);
        f3gbm_train_inner();
    end
    
    if mod(epoch, gbm.everySave) == 0 || epoch == gbm.n_epoch
        if gbm.saveFile
            fileName = [gbm.datestring '_' int2str(epoch) '.mat'];
            save(fileName,'gbm','epoch');
        end
    end
    
end

    function f3gbm_train_inner()
        delta = gbm.momentum*delta - gbm.stepsize*f3gbm_grad();
        
        fprintf('epoch %d ## norm of inc: %f\n', epoch ,norm(delta));
        
        % stablize the things
        if norm(delta) > gbm.deltaMax
            delta = delta/norm(delta)*deltaMax;
            fprintf('epoch %d ## norm of inc again: %f\n', epoch ,norm(delta));
        end
        
        assert(all(~isnan(delta)));
        
        %add separately
        gbm.Wxf(:) = gbm.Wxf(:) + delta(d_pos(1)+1:d_pos(2));
        gbm.Wyf(:) = gbm.Wyf(:) + delta(d_pos(2)+1:d_pos(3));
        gbm.Whf(:) = gbm.Whf(:) + delta(d_pos(3)+1:d_pos(4));
        gbm.wx(:)  = gbm.wx(:)  + delta(d_pos(4)+1:d_pos(5));
        gbm.wy(:)  = gbm.wy(:)  + delta(d_pos(5)+1:d_pos(6));
        gbm.wh(:)  = gbm.wh(:)  + delta(d_pos(6)+1:d_pos(7));
        
        function grad = f3gbm_grad()
            % contrastive divergence approximation
            X = batch_X;
            Y = batch_Y;
            H = sigm((X*gbm.Wxf).*(Y*gbm.Wyf)*(gbm.Whf)'+ones(gbm.batchsize,1)*gbm.wh');
            g_Wxf = -X'*((Y*gbm.Wyf).*(H*gbm.Whf))/gbm.batchsize;
            g_Wyf = -Y'*((X*gbm.Wxf).*(H*gbm.Whf))/gbm.batchsize;
            g_Whf = -H'*((X*gbm.Wxf).*(Y*gbm.Wyf))/gbm.batchsize;
            g_wx  = -mean(X,1)';
            g_wy  = -mean(Y,1)';
            g_wh  = -mean(H,1)';
            H = double(H > rand(size(H)));
            
            for iCD=1:gbm.cditerations
                if ~gbm.meanfield_output
                    if isequal(gbm.visType, 'binary')
                        X = double(sigm((Y*gbm.Wyf).*(H*gbm.Whf)*(gbm.Wxf)'+ones(gbm.batchsize,1)*gbm.wx') > rand(size(X)));
                        Y = double(sigm((X*gbm.Wxf).*(H*gbm.Whf)*(gbm.Wyf)'+ones(gbm.batchsize,1)*gbm.wy') > rand(size(Y)));
                    else
                        X = (Y*gbm.Wyf).*(H*gbm.Whf)*(gbm.Wxf)'+ones(gbm.batchsize,1)*gbm.wx' + randn(size(X));
                        Y = (X*gbm.Wxf).*(H*gbm.Whf)*(gbm.Wyf)'+ones(gbm.batchsize,1)*gbm.wy' + randn(size(Y));
                    end
                else
                    if isequal(gbm.visType, 'binary')
                        X = sigm((Y*gbm.Wyf).*(H*gbm.Whf)*(gbm.Wxf)'+ones(gbm.batchsize,1)*gbm.wx');
                        Y = sigm((X*gbm.Wxf).*(H*gbm.Whf)*(gbm.Wyf)'+ones(gbm.batchsize,1)*gbm.wy');
                    else
                        X = (Y*gbm.Wyf).*(H*gbm.Whf)*(gbm.Wxf)'+ones(gbm.batchsize,1)*gbm.wx';
                        Y = (X*gbm.Wxf).*(H*gbm.Whf)*(gbm.Wyf)'+ones(gbm.batchsize,1)*gbm.wy';
                    end
                end
                H = sigm((X*gbm.Wxf).*(Y*gbm.Wyf)*(gbm.Whf)'+ones(gbm.batchsize,1)*gbm.wh');
            end
            
            g_Wxf = g_Wxf + X'*((Y*gbm.Wyf).*(H*gbm.Whf))/gbm.batchsize;
            g_Wyf = g_Wyf + Y'*((X*gbm.Wxf).*(H*gbm.Whf))/gbm.batchsize;
            g_Whf = g_Whf + H'*((X*gbm.Wxf).*(Y*gbm.Wyf))/gbm.batchsize;
            g_wx  = g_wx + mean(X,1)';
            g_wy  = g_wy + mean(Y,1)';
            g_wh  = g_wh + mean(H,1)';
            H = double(H > rand(size(H)));
            
            grad = [g_Wxf(:);g_Wyf(:);g_Whf(:);g_wx(:);g_wy(:);g_wh(:)];
        end
    end

end

