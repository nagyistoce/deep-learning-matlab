load('test_f4_data.mat');

f4gbm_test(gbm, test_x, test_y, test_z, 'gibbs_steps', 100, 'show', true, 'random_init', false);