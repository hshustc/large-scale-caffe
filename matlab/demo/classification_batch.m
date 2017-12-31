% ------------------------------------------------------------------------
function [scores, maxlabel] = classification_batch(list_im, gpu_id, model_def_file, model_file, mean_file, dim, batch_size, crop_size)
% ------------------------------------------------------------------------
% function [scores, maxlabel] = classification_batch(list_im, gpu_id, model_def_file, model_file, mean_file, dim, batch_size, crop_size)
    if exist('../+caffe', 'dir')
        addpath('..');
    else
        error('Please run this demo from caffe/matlab/demo');
    end

    if gpu_id >= 0
        caffe.set_mode_gpu();
        caffe.set_device(gpu_id);
    else
        caffe.set_mode_cpu()
    end


    % Initialize the network using BVLC CaffeNet for image classification
    % Weights (parameter) file needs to be downloaded from Model Zoo.
    % model_dir = '../../models/bvlc_reference_caffenet/';
    % net_model = [model_dir 'deploy.prototxt'];
    % net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];
    net_model = model_def_file;
    net_weights = model_file;
    phase = 'test'; % run with phase test (so that dropout isn't applied)
    % if ~exist(net_weights, 'file')
    %   error('Please download CaffeNet from Model Zoo before you run this demo');
    % end
    if ~exist(net_model, 'file')
        error(sprintf('%s is missing\n', net_model));
    end
    if ~exist(net_weights, 'file')
        error(sprintf('%s is missing\n', net_weights));
    end

    % Initialize a network
    net = caffe.Net(net_model, net_weights, phase);

    % prepare oversampled input
    % input_data is Height x Width x Channel x Num
    % tic;
    % input_data = {prepare_image(im)};
    % toc;
    % 
    % % do forward pass to get scores
    % % scores are now Channels x Num, where Channels == 1000
    % tic;
    % % The net forward function. It takes in a cell array of N-D arrays
    % % (where N == 4 here) containing data of input blob(s) and outputs a cell
    % % array containing data from output blob(s)
    % scores = net.forward(input_data);
    % toc;
    % 
    % scores = scores{1};
    % scores = mean(scores, 2);  % take average scores over 10 crops
    % 
    % [~, maxlabel] = max(scores);
    num_images = length(list_im);
    scores = zeros(dim,num_images,'single');
    num_batches = ceil(length(list_im)/batch_size)
    initic=tic;
    for bb = 1:num_batches
        batchtic = tic;
        range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
        tic
        input_data = prepare_batch(list_im(range),mean_file,batch_size, crop_size);
        toc, tic
        fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
        %output_data = caffe('forward', {input_data});
        output_data = net.forward({input_data});
        toc
        output_data = squeeze(output_data{1});
        scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
        toc(batchtic)
    end
    toc(initic);

    [~, maxlabel] = max(scores);

    % call caffe.reset_all() to reset caffe
    caffe.reset_all();

end %end classification_batch

% ------------------------------------------------------------------------
function crops_data = prepare_batch(image_files, mean_file, batch_size, crop_size)
% ------------------------------------------------------------------------
    % caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
    % is already in W x H x C with BGR channels
    % d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
    d = load(mean_file);
    mean_data = d.mean_data;
    IMAGE_DIM = 256;
    CROPPED_DIM = crop_size;

    num_images = length(image_files);
    crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, batch_size, 'single');
    indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
    center = floor(indices(2) / 2)+1;

    % parfor i=1:num_images
    for i=1:num_images
        % fprintf('Preparing %s\n', image_files{i});
        try
            im = imread(image_files{i});
            % Convert an image returned by Matlab's imread to im_data in caffe's data
            % format: W x H x C with BGR channels
            im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
            im_data = permute(im_data, [2, 1, 3]);  % flip width and height
            im_data = single(im_data);  % convert from uint8 to single
            im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
            im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

            crops_data(:, :, :, i) = im_data(center:center+CROPPED_DIM-1, center:center+CROPPED_DIM-1, :);

            % oversample (4 corners, center, and their x-axis flips)
            % crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
            % indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
            % n = 1;
            % for i = indices
            %     for j = indices
            %         crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
            %         crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
            %         n = n + 1;
            %   end
            % end
            % center = floor(indices(2) / 2) + 1;
            % crops_data(:,:,:,5) = ...
            % im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
            % crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);
        catch
            warning('Problem with file', image_files{i});
            % error(sprintf('Problem with file: %s\n', image_files{i}));
        end
    end

end %end prepare_batch

