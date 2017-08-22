from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData
from cntk.core import Value
from od_reader import ObjectDetectionReader
import numpy as np

class ObjectDetectionMinibatchSource(UserMinibatchSource):
    def __init__(self, url , max_annotations_per_image,
                 pad_width, pad_height, pad_value, randomize, use_flipping,
                 max_images=None):

        self.image_si = StreamInformation("image", 0, 'dense', np.float32, (3, pad_height, pad_width,))
        self.roi_si = StreamInformation("annotation", 1, 'dense', np.float32, (max_annotations_per_image, 5,))
        self.dims_si = StreamInformation("dims", 1, 'dense', np.float32, (4,))

        self.od_reader = ObjectDetectionReader(url ,max_annotations_per_image,
                 pad_width, pad_height, pad_value, randomize, use_flipping, max_images)

        super(ObjectDetectionMinibatchSource, self).__init__()

    def stream_infos(self):
        return [self.image_si, self.roi_si, self.dims_si]

    def image_si(self):
        return self.image_si

    def roi_si(self):
        return self.roi_si

    def dims_si(self):
        return self.dims_si

    def next_minibatch(self, url, num_samples, number_of_workers=1, worker_rank=1, device=None, input_map=None):
        img_data, roi_data, img_dims = self.od_reader.get_next_input(url)
        sweep_end = self.od_reader.sweep_end()

        if input_map is None:
            result = {
                self.image_si: MinibatchData(Value(batch=img_data), 1, 1, sweep_end),
                self.roi_si:   MinibatchData(Value(batch=roi_data), 1, 1, sweep_end),
                self.dims_si:  MinibatchData(Value(batch=np.asarray(img_dims, dtype=np.float32)), 1, 1, sweep_end),
            }
        else:
            result = {
                input_map[self.image_si]: MinibatchData(Value(batch=np.asarray(img_data, dtype=np.float32)), 1, 1, sweep_end),
                input_map[self.roi_si]:   MinibatchData(Value(batch=np.asarray(roi_data, dtype=np.float32)), 1, 1, sweep_end),
                input_map[self.dims_si]:  MinibatchData(Value(batch=np.asarray(img_dims, dtype=np.float32)), 1, 1, sweep_end),
            }
        return result 
