clear all;
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
mean_path = '../../fine-grained_data/birds_box/data/';
mean_name = 'birds_box_mean';
%image_mean = caffe('read_mean', [mean_path, mean_name, '.binaryproto']);
%image_mean = permute(image_mean, [2,1,3]);
mean_data = caffe.io.read_mean([mean_path, mean_name, '.binaryproto']);
save([mean_path, mean_name, '.mat'], 'mean_data');
