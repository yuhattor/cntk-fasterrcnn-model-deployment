import zipfile
import cv2 # pip install opencv-python
import numpy as np
import os
import pdb
import urllib.request as ur



class ObjectDetectionReader:
    def __init__(self, url, max_annotations_per_image,
                 pad_width, pad_height, pad_value, randomize, use_flipping,
                 max_images=None):
        self._pad_width = pad_width
        self._pad_height = pad_height
        self._pad_value = pad_value
        self._randomize = randomize
        self._use_flipping = use_flipping
        self._flip_image = True # will be set to False in the first call to _reset_reading_order
        self._img_file_paths = []
        self._gt_annotations = []
        self._num_images = 1 #self._parse_map_files()
        annotations = np.zeros((50, 5))
        self._gt_annotations.append(annotations)
        self._img_stats = [None for _ in range(self._num_images)]
        self._reading_order = None
        self._reading_index = -1
        
    def get_next_input(self, url):
        index = self._get_next_image_index()
        roi_data = self._get_gt_annotations(index)
        img_data, img_dims = self._load_resize_and_pad_image(url)
        return img_data, roi_data, img_dims 

    def sweep_end(self):
        return self._reading_index >= self._num_images


    def _reset_reading_order(self):
        self._reading_order = np.arange(self._num_images)
        if self._randomize:
            np.random.shuffle(self._reading_order)
        # if flipping should be used then we alternate between epochs from flipped to non-flipped
        self._flip_image = not self._flip_image if self._use_flipping else False
        self._reading_index = 0

    def _prepare_annotations_and_image_stats(self, index, img_width, img_height):
        annotations = self._gt_annotations[index]
        do_scale_w = img_width > img_height
        target_w = self._pad_width
        target_h = self._pad_height

        if do_scale_w:
            scale_factor = float(self._pad_width) / float(img_width)
            target_h = int(np.round(img_height * scale_factor))
        else:
            scale_factor = float(self._pad_height) / float(img_height)
            target_w = int(np.round(img_width * scale_factor))

        top = int(max(0, np.round((self._pad_height - target_h) / 2)))
        left = int(max(0, np.round((self._pad_width - target_w) / 2)))
        bottom = self._pad_height - top - target_h
        right = self._pad_width - left - target_w

        xyxy = annotations[:, :4]
        xyxy *= scale_factor
        xyxy += (left, top, left, top)

        # not needed since xyxy is just a reference: annotations[:, :4] = xyxy
        # TODO: do we need to round/floor/ceil xyxy coords?
        annotations[:, 0] = np.round(annotations[:, 0])
        annotations[:, 1] = np.round(annotations[:, 1])
        annotations[:, 2] = np.round(annotations[:, 2])
        annotations[:, 3] = np.round(annotations[:, 3])

        # keep image stats for scaling and padding images later
        img_stats = [target_w, target_h, img_width, img_height, top, bottom, left, right]
        self._img_stats[index] = img_stats

    def _get_next_image_index(self):
        if self._reading_index < 0 or self._reading_index >= self._num_images:
            self._reset_reading_order()
        next_image_index = self._reading_order[self._reading_index]
        self._reading_index += 1
        return next_image_index

    def _load_resize_and_pad_image(self, url, index = 0):
        #image_path = self._img_file_paths[index]
        resp = ur.urlopen(url)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if self._img_stats[index] is None:
            img_width = len(img[0])
            img_height = len(img)
            self._prepare_annotations_and_image_stats(index, img_width, img_height)

        target_w, target_h, img_width, img_height, top, bottom, left, right = self._img_stats[index]

        resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)
        resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                              value=self._pad_value)
        if self._flip_image:
            resized_with_pad = cv2.flip(resized_with_pad, 1)

        # transpose(2,0,1) converts the image to the HWC format which CNTK accepts
        model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

        # dims = pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height
        dims = (self._pad_width, self._pad_height, target_w, target_h, img_width, img_height)
        return model_arg_rep, dims

    def _get_gt_annotations(self, index):
        annotations = self._gt_annotations[index]
        if self._flip_image:
            flipped_annotations = np.array(annotations)
            flipped_annotations[:,0] = self._pad_width - annotations[:,2] - 1
            flipped_annotations[:,2] = self._pad_width - annotations[:,0] - 1
            return flipped_annotations
        return annotations

