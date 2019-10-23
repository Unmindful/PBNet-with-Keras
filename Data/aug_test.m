
dataDir = 'Validation Data/Set14';
folder=fullfile('Validation Data','val_Set14_128x128bicY');
mkdir(folder);
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];

for f_iter = 1:numel(f_lst)

    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    disp(f_path);
    img_raw = imread(f_path);
    
    if size(img_raw,3)~=3
        img_raw = repmat(img_raw, [1 1 3]);
    end
    
    if size(img_raw,3)==3
        img_raw = rgb2ycbcr(img_raw);
        img_raw = img_raw(:,:,1);
    else
        img_raw = rgb2ycbcr(repmat(img_raw, [1 1 3]));
    end
        
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    patch_size = 128;
    stride = 128;    
    img_raw = img_raw(1:height-mod(height,patch_size),1:width-mod(width,patch_size),:);
    img_raw = im2double(img_raw);
    
    img_size = size(img_raw);

    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    
    img_2 = imresize(imresize(img_raw,1/2,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_3 = imresize(imresize(img_raw,1/3,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_4 = imresize(imresize(img_raw,1/4,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    img_8 = imresize(imresize(img_raw,1/8,'bicubic'),[img_size(1),img_size(2)],'bicubic');
    
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
                       
            % Actual Alignment of image
            
            patch_name = sprintf('%s/%d',folder,count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_4', patch_name), 'patch');
            patch = imrotate(img_8(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            save(sprintf('%s_8', patch_name), 'patch');
            
            count = count + 1;
            
        end
    end
    
    display(count);
    
    
end
