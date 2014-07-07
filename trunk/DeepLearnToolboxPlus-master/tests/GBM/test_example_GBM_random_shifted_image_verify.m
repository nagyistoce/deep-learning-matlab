function test_example_GBM_random_shifted_image_verify(fileName)
% TEST_EXAMPLE_GBM_RANDOM_SHIFTED_IMAGE_VERIFY ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 16-May-2014 19:11:25 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : test_example_GBM_random_shifted_image_verify.m 

% generate test data 

if nargin < 1 || isempty(fileName)
    fileName = 'gbm_random_shifted_image_20140519T131250.mat';
end

gbmStruct = load(fileName);

sideSize = 10;
probOn = 0.1;
debug = false;
N = 50000;

% [xDataTest, yDataTest] = gbm_generate_random_shifted_image(N,sideSize,probOn,debug);

dataTrain = load('gbm_image_data.mat');

xDataTest = dataTrain.xData;
yDataTest = dataTrain.yData;

overallWeight = gbmStruct.gbm.W_xyh;
h1_all = 0;
for iTestImage = 1:N
    xImageThis = xDataTest(iTestImage,:);
    yImageThis = yDataTest(iTestImage,:);
    
    if isequal(xImageThis,yImageThis)
        continue; % need different images.
    end
%     which gbm
%     which gbm -all
%     gbm.xSize;
%     disp(gbm);

    [~,~,h1] = bm.gbm_statistics_single(xImageThis,yImageThis,gbmStruct.gbm,'CD',true);
    
%     h1_all = h1+h1_all;
    
    h1
    marginalWeight = mean( bsxfun(@times,overallWeight,reshape(h1,[1,1,10])),3);
    [~,reconstructedY] = max(marginalWeight,[],2);
    
    reconstructedYImage = zeros(1,sideSize*sideSize);
    
    for iPixel = 1:sideSize*sideSize
        reconstructedYImage(reconstructedY(iPixel)) = xImageThis(iPixel);
    end
    
    close all;
    
    subplot(3,1,1);
    imagesc(reshape(xImageThis,sideSize,sideSize)); colormap gray;
    subplot(3,1,2);
    imagesc(reshape(yImageThis,sideSize,sideSize)); colormap gray;
    subplot(3,1,3);
    imagesc(reshape(reconstructedYImage,sideSize,sideSize)); colormap gray;
    
    fprintf('error rate: %d/%d = %f\n', sum(reconstructedYImage~=yImageThis),sideSize*sideSize, sum(reconstructedYImage~=yImageThis)/sideSize*sideSize);
    
    
    % flow field
    
    [reconstructedY2D1,reconstructedY2D2] = ind2sub([sideSize,sideSize],reconstructedY);
    
    [gx,gy] = meshgrid(1:sideSize,1:sideSize);
    u = zeros(size(gx));
    v = zeros(size(gy));
    
    
    for iPixel = 1:numel(gx,gy)
        u(iPixel) = reconstructedY2D2(iPixel)-gx(iPixel);
        v(iPixel) = -(reconstructedY2D1(iPixel)-gy(iPixel));
    end
    
    
    figure; 
    quiver(gx,gy,u,v);
    
    
    
    pause;
%     disp(iTestImage);
    
end
% h1_all/N

end








% Created with NEWFCN.m by Frank Gonz�lez-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [test_example_GBM_random_shifted_image_verify.m] ======  
