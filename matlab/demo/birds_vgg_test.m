%birds_vgg_test
exists_or_mkdir('./birds');
root_path = './birds';

list_im_file= fullfile(root_path, 'birds_test.txt');
mean_file = fullfile(root_path, 'birds_mean.mat');
dim = 200;
image_path = '/home/vim/fine-grained_data/birds/data/birds_images/';
batch_size = 32;
crop_size = 224;
gpu_id = 1;

deploy_file =  'vgg_deploy.prototxt';
model_def_file = fullfile('/home/vim/fine-grained_data/birds/model/v4_vggnet', deploy_file);

snapshot = 'birds_vgg_train_iter_60000.caffemodel';
model_file = fullfile('/home/vim/fine-grained_data/birds/model/v4_vggnet/snapshot', snapshot);

fid = fopen(list_im_file, 'r');
im_label = textscan(fid, '%s %s');
fclose(fid);
list_im = im_label{1};
list_label = im_label{2};
gt_labels = zeros(1, length(list_im));

for i=1:length(list_im)
    list_im{i} = [image_path, list_im{i}];
    gt_labels(i) = str2num(list_label{i}) + 1;
end

[scores, maxlabel] = classification_batch(list_im, gpu_id,  model_def_file, model_file, mean_file, dim, batch_size, crop_size);
assert(size(scores, 1) == dim);
assert(size(scores, 2) == length(list_im));

if ~isempty(strfind(model_def_file, 'fine'))
    result_file = ['birds_vgg_test_result_fine_', snapshot, '.mat'];
elseif ~isempty(strfind(model_def_file, 'coarse'))
    result_file = ['birds_vgg_test_result_coarse_', snapshot, '.mat'];
elseif ~isempty(strfind(model_def_file, 'combine'))
    result_file = ['birds_vgg_test_result_combine_', snapshot, '.mat'];
elseif ~isempty(strfind(model_def_file, 'fused')) %fused means the Fused Classifier
    result_file = ['birds_vgg_test_result_fused_', snapshot, '.mat']; 
else
    result_file = ['birds_vgg_test_result_', snapshot, '.mat'];
end

result_file = fullfile(root_path, result_file);

num_correct = length(find(gt_labels == maxlabel));
meanacc = num_correct ./ length(gt_labels)
save(result_file, 'maxlabel', 'scores', 'meanacc', '-v7.3');
