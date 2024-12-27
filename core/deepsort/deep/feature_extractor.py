import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        inps = []
        for im in im_crops:
            _im = im.astype(np.float32) / 255
            _im = cv2.resize(_im, self.size)
            _im = self.norm(_im)
            _im = _im.unsqueeze(0)
            inps.append(_im)

        im_batch = torch.cat(inps, dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("data/images/army.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("weights/deepsort/ckpt.t7")
    feature = extr([img])
    print(feature.shape)
