load('train_f4_data.mat');

gbm = f4gbm_setup(100, 100, 100, 150, 500,...
'stepsize',0.01,'meanfield_output',false,...
'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
'weightPenaltyL1',0,...
'everySave',200,'n_epoch',400,'visType','binary','saveFile',true,'seed',0);

gbm = f4gbm_train(gbm, train_x, train_y, train_z);

figure,
visualize(gbm.wxf);
figure,
visualize(gbm.wyf);
figure,
visualize(gbm.wzf);