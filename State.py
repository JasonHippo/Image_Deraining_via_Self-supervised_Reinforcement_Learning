import numpy as np
import cv2

class State():
    def __init__(self, move_range):
        self.move_range = move_range
    
    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:,:self.image.shape[1],:,:] = self.image

    def step(self, act, inner_state):
        neutral = (self.move_range - 1)/2
        move = act.astype(np.float32)
        move = (move - neutral)/255
        moved_image = self.image + move[:,np.newaxis,:,:]
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==self.move_range) > 0: 
                gaussian[i] = cv2.GaussianBlur(self.image[i].transpose([1, 2, 0]), ksize=(5,5), sigmaX=0.5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+1) > 0: 
                bilateral[i] = cv2.bilateralFilter(self.image[i].transpose([1, 2, 0]), d=5, sigmaColor=0.1, sigmaSpace=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+2) > 0:
                median[i] = cv2.medianBlur(self.image[i].transpose([1, 2, 0]), ksize=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+3) > 0: 
                gaussian2[i] = cv2.GaussianBlur(self.image[i].transpose([1, 2, 0]), ksize=(5,5), sigmaX=1.5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+4) > 0: 
                bilateral2[i] = cv2.bilateralFilter(self.image[i].transpose([1, 2, 0]), d=5, sigmaColor=1.0, sigmaSpace=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+5) > 0: 
                box[i] = cv2.boxFilter(self.image[i].transpose([1, 2, 0]), ddepth=-1, ksize=(5,5)).transpose([2, 0, 1])
        self.image = moved_image
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range, gaussian, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+1, bilateral, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+2, median, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+3, gaussian2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+4, bilateral2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+5, box, self.image)
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state

    def step_with_mask(self, act, mask, inner_state):
        neutral = (self.move_range - 1)/2
        mask = np.squeeze(mask, axis=1)
        act = np.where(mask==0, 1, act) # if mask equal to 0, the pixel isn't a rain, so the act should be id==1:"do nothing" 
        move = act.astype(np.float32)
        move = (move - neutral)/255
        moved_image = self.image + move[:,np.newaxis,:,:]
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==self.move_range) > 0:
                gaussian[i] = cv2.GaussianBlur(self.image[i].transpose([1, 2, 0]), ksize=(5,5), sigmaX=0.5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+1) > 0:
                bilateral[i] = cv2.bilateralFilter(self.image[i].transpose([1, 2, 0]), d=5, sigmaColor=0.1, sigmaSpace=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+2) > 0: 
                median[i] = cv2.medianBlur(self.image[i].transpose([1, 2, 0]), ksize=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+3) > 0: 
                gaussian2[i] = cv2.GaussianBlur(self.image[i].transpose([1, 2, 0]), ksize=(5,5), sigmaX=1.5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+4) > 0: 
                bilateral2[i] = cv2.bilateralFilter(self.image[i].transpose([1, 2, 0]), d=5, sigmaColor=1.0, sigmaSpace=5).transpose([2, 0, 1])
            if np.sum(act[i]==self.move_range+5) > 0: 
                box[i] = cv2.boxFilter(self.image[i].transpose([1, 2, 0]), ddepth=-1, ksize=(5,5)).transpose([2, 0, 1])
        self.image = moved_image
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range, gaussian, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+1, bilateral, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+2, median, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+3, gaussian2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+4, bilateral2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+5, box, self.image)
        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state