patchsize = 100;
n_patch   = 10000;
p_one = 0.05;

len = sqrt(patchsize);
train_x = zeros(n_patch, patchsize);
train_y = zeros(n_patch, patchsize);
train_z = zeros(n_patch, patchsize);

tmp_y = zeros(len);
for i=1:n_patch
    tmp_y(2:len-1,2:len-1) = double(rand(len-2) < p_one);
    tmp = randi(5);
    tmp_x = zeros(len);
    tmp_z = zeros(len);
    switch tmp
        case 1
            tmp_x = tmp_y;
            tmp_z = tmp_y;
        case 2
            tmp_x(1:len-2,2:len-1) = tmp_y(2:len-1,2:len-1);
            tmp_z(3:len,2:len-1) = tmp_y(2:len-1,2:len-1);
        case 3
            tmp_x(3:len,2:len-1) = tmp_y(2:len-1,2:len-1);
            tmp_z(1:len-2,2:len-1) = tmp_y(2:len-1,2:len-1);
        case 4
            tmp_x(2:len-1,1:len-2) = tmp_y(2:len-1,2:len-1);
            tmp_z(2:len-1,3:len) = tmp_y(2:len-1,2:len-1);
        case 5
            tmp_x(2:len-1,3:len) = tmp_y(2:len-1,2:len-1);
            tmp_z(2:len-1,1:len-2) = tmp_y(2:len-1,2:len-1);
    end
    train_x(i,:) = tmp_x(:)';
    train_y(i,:) = tmp_y(:)';
    train_z(i,:) = tmp_z(:)';
end

figure,
subplot(1,3,1), imshow(imresize(reshape(train_x(1,:), [len, len]),20));
subplot(1,3,2), imshow(imresize(reshape(train_y(1,:), [len, len]),20));
subplot(1,3,3), imshow(imresize(reshape(train_z(1,:), [len, len]),20));

save train_f4_data.mat train_x train_y train_z