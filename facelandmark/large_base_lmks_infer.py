import torch
import numpy as np
from facelandmark.nets.large_base_lmks_net import LargeBaseLmksNet
from util.device import get_torch_device, load_checkpoint

BASE_LANDMARK_NUM = 106
INPUT_SIZE = 224
ENLARGE_RATIO = 1.35


class LargeBaseLmkInfer:
    @staticmethod
    def model_preload(model_path, device=None):
        device = get_torch_device() if device is None else torch.device(device)
        model = LargeBaseLmksNet(infer=False)
        model = model.to(device)

        checkpoint = load_checkpoint(model_path, device)

        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=False)
        model.eval()
        return model

    @staticmethod
    def process_img(model, image, device=None):
        device = get_torch_device() if device is None else torch.device(device)
        img_resize = image

        # cv2.imshow("test", img_resize.astype(np.uint8))
        # cv2.waitKey()

        img_resize = ((img_resize - [103.94, 116.78, 123.68]) / 255.0).astype(np.float32)  # important
        img_resize = img_resize.transpose([2, 0, 1])

        img_resize = torch.from_numpy(img_resize).to(device)

        w_new = INPUT_SIZE
        h_new = INPUT_SIZE
        img_in = torch.zeros([1, 3, h_new, w_new], dtype=torch.float32, device=device)

        img_in[0, :] = img_resize

        with torch.no_grad():
            output = model(img_in)
            output = output * INPUT_SIZE

        output = output.cpu().numpy()

        return output

    @staticmethod
    def smooth(cur_lmks, prev_lmks):
        smooth_lmks = np.zeros((106, 2))

        cur_rect_x1 = np.min(cur_lmks[:, 0])
        cur_rect_x2 = np.max(cur_lmks[:, 0])

        smooth_param = 60.0
        factor = smooth_param / (cur_rect_x1 - cur_rect_x2)
        for i in range(BASE_LANDMARK_NUM):
            weightX = np.exp(factor * np.abs(cur_lmks[i][0] - prev_lmks[i][0]))
            weightY = np.exp(factor * np.abs(cur_lmks[i][1] - prev_lmks[i][1]))

            smooth_lmks[i][0] = (1 - weightX) * cur_lmks[i][0] + weightX * prev_lmks[i][0]
            smooth_lmks[i][1] = (1 - weightY) * cur_lmks[i][1] + weightY * prev_lmks[i][1]

        return smooth_lmks

    @staticmethod
    def infer_img(img, model, device=None):
        lmks = LargeBaseLmkInfer.process_img(model, img, device)
        return lmks
