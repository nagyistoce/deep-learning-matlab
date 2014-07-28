load('modelInitAndData.mat');

gbm = f3gbm_setup(169, 169, 100, 200,...
'stepsize',0.01,'meanfield_output',false,...
'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0.0001,...
'weightPenaltyL1',0,...
'everySave',200,'n_epoch',600,'visType','binary','saveFile',true,'seed',0);

gbm = f3gbm_train(gbm, inputimages, outputimages);

figure,
visualize(gbm.wxf);
figure,
visualize(gbm.wyf);