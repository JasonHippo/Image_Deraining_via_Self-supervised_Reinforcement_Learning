import os
import numpy as np
import cv2
import random

class MiniBatchLoader(object):
 
    def __init__(self, train_path, image_dir_path):
        self.training_path_infos = self.read_paths(train_path, image_dir_path)

    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, index):
        return self.load_data(self.training_path_infos, index)
 
    def load_data(self, path_infos, index):
        in_channels = 3
        path = path_infos[index]
        mask_path = path.replace('input', 'RDP5')
        pseudo_gt_dir = path.replace('input', 'Pseudo_Reference_RDP5')[:-4]

        img = cv2.imread(path)
        if '.jpg' in path:
            mask = cv2.imread(mask_path.replace('.jpg', '.png'), 0)
        else:
            mask = cv2.imread(mask_path, 0)
        if img is None or mask is None:
            raise RuntimeError("invalid image: {i}".format(i=path))
        h, w, c = img.shape
        pseudo_ys = np.zeros((50, in_channels, h, w)).astype(np.float32)
        for i in range(0, 50):
            pseudo_gt = cv2.imread(os.path.join(pseudo_gt_dir, '{}.png'.format(i)))
            pseudo_ys[i] = (pseudo_gt/255).transpose([2, 0, 1]).astype(np.float32)

        xs = np.zeros((1, in_channels, h, w)).astype(np.float32)
        masks = np.zeros((1, 1, h, w)).astype(np.float32)
        xs[0] = (img/255).transpose([2, 0, 1]).astype(np.float32)
        masks[0, 0, :, :] = (mask/255).astype(np.float32)
        
        return xs, pseudo_ys, masks, os.path.basename(path)
    
    def load_training_batch_data(self, indices, img_size):
        return self.load_batch_data(self.training_path_infos, indices, img_size, augment=True)
    
    def load_testing_batch_data(self, indices):
        return self.load_batch_data(self.training_path_infos, indices, None, augment=False)
     
    def load_batch_data(self, path_infos, indices, img_size=None, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3

        if augment:
            assert img_size != None, "Request the parameter of [img_size] should not be None."
            
            xs = np.zeros((mini_batch_size, in_channels, img_size, img_size)).astype(np.float32)
            pseudo_ys = np.zeros((mini_batch_size, in_channels, img_size, img_size)).astype(np.float32)
            masks = np.zeros((mini_batch_size, 1, img_size, img_size)).astype(np.float32)
            
            for i, index in enumerate(indices):
                path = path_infos[index]
                mask_path = path.replace('input', 'RDP5')
                pseudo_gt_dir = path.replace('input', 'Pseudo_Reference_RDP5')[:-4]

                img = cv2.imread(path)
                if '.jpg' in path:
                    mask = cv2.imread(mask_path.replace('.jpg', '.png'), 0)
                else:
                    mask = cv2.imread(mask_path, 0)
                    
                if img is None or mask is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                
                random_pseudo_gt_index = random.randint(0, 49)
                pseudo_gt = cv2.imread(os.path.join(pseudo_gt_dir, '{}.png'.format(random_pseudo_gt_index)))

                if pseudo_gt is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                
                h, w, c = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                    pseudo_gt = np.fliplr(pseudo_gt)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    img = cv2.warpAffine(img,M,(w,h))
                    mask = cv2.warpAffine(mask,M,(w,h))
                    pseudo_gt = cv2.warpAffine(pseudo_gt,M,(w,h))

                rand_range_h = h-img_size
                rand_range_w = w-img_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = img[y_offset:y_offset+img_size, x_offset:x_offset+img_size, :]    
                mask = mask[y_offset:y_offset+img_size, x_offset:x_offset+img_size]
                pseudo_gt = pseudo_gt[y_offset:y_offset+img_size, x_offset:x_offset+img_size, :]
                xs[i] = (img/255).transpose([2, 0, 1]).astype(np.float32)
                pseudo_ys[i] = (pseudo_gt/255).transpose([2, 0, 1]).astype(np.float32)
                masks[i, 0, :, :] = (mask/255).astype(np.float32)

            return xs, pseudo_ys, masks
            
        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                mask_path = path.replace('input', 'RDP5')

                img = cv2.imread(path)
                if '.jpg' in path:
                    mask = cv2.imread(mask_path.replace('.jpg', '.png'), 0)
                else:
                    mask = cv2.imread(mask_path, 0)
                    
                if img is None or mask is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                
            h, w, c = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            masks = np.zeros((mini_batch_size, 1, h, w)).astype(np.float32)
            xs[0] = (img/255).transpose([2, 0, 1]).astype(np.float32)
            masks[0, 0, :, :] = (mask/255).astype(np.float32)

            return xs, masks, os.path.basename(path)
            
        else:
            raise RuntimeError("mini batch size must be 1 when testing")