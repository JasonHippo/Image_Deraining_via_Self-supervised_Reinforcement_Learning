clc;close all;clear all;

% input file path
file_path = strcat('../dataset/Rain100L/test/input/');
% save file path
rain_component_path = strcat('dataset/Rain100L/test/RDP_before_binarization/');

if ~exist(rain_component_path, 'dir')
   mkdir(rain_component_path)
end

path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
img_num = length(path_list);
if img_num > 0
    for i = 1:img_num 
        image_name = path_list(i).name;
        I = imread(strcat(file_path,image_name));
        I = rgb2gray(I);
        I = double(I) / 255;
        ILF = bfilter2(I, 5, [6, 0.2]);
        Ori_IHF = I - ILF;
        [NonRainComponent, RainComponent, Texture_Dict, Cartoon_Dict] = MCA_Image_Decomposition(Ori_IHF, 16, 100, 1024, [], [], 3);
        if endsWith(image_name, '.jpg')
            imwrite(uint8(RainComponent*255), strcat(rain_component_path, strrep(image_name, '.jpg', '.png')));
            fprintf(1, 'Save %s!\n', strrep(image_name, '.jpg', '.png'));
        else
            imwrite(uint8(RainComponent*255), strcat(rain_component_path, image_name));
            fprintf(1, 'Save %s!\n', image_name);
        end
    end
end