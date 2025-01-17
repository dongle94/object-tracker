import numpy as np

from deep.feature_extractor import Extractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker
from utils.logger import get_logger


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, nn_budget=100, use_cuda=True):

        self.logger = get_logger()
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        print(f"Loading weights from {model_path}... Done!")

        self.max_cosine_distance = max_dist
        self.nn_budget = nn_budget
        metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)

        self.width = None
        self.height = None

    def update(self, dets, ori_img):
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        bbox_xyxy = dets[:, :4].copy()
        scores = dets[:, 4]
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(scores)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = map(int, box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        bbox_xyxy[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_xyxy


    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2


if __name__ == "__main__":
    import cv2
    from core.obj_detector import ObjectDetector
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from utils.visualize import get_color_for_id

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _tracker = DeepSort(
        model_path='./weights/deepsort/ckpt.t7',
        max_dist=0.2,
        nn_budget=100,
        use_cuda=True
    )

    _video_capture = cv2.VideoCapture('/home/dongle94/Videos/mot/TUD-Stadtmitte-raw.webm')

    ts = [0., 0., 0.]
    f = 0
    while _video_capture.isOpened():
        f += 1
        t0 = _detector.detector.get_time()
        _ret, _img = _video_capture.read()
        if _img is None:
            break
        t1 = _detector.detector.get_time()

        _det = _detector.run(_img)
        t2 = _detector.detector.get_time()

        # _tracker.update
        _tracklet = _tracker.update(_det, _img)
        t3 = _detector.detector.get_time()

        for _d in _det:
            _x1, _y1, _x2, _y2 = map(int, _d[:4])
            cls = int(_d[5])
            cv2.rectangle(_img, (_x1, _y1), (_x2, _y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(_img, str(_detector.names[cls]), (_x1, _y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

        # tracking update
        if _tracklet:
            for t in _tracklet:
                _x1, _y1, _x2, _y2 = map(int, t[:4])
                _tracking_id = int(t[4])
                color = get_color_for_id(_tracking_id)
                cv2.rectangle(_img, (_x1, _y1), (_x2, _y2), color, thickness=2)
                cv2.putText(_img, f"ID: {_tracking_id}", (_x1, _y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

        ts[0] += t1 - t0
        ts[1] += t2 - t1
        ts[2] += t3 - t2

        cv2.imshow('_', _img)
        if cv2.waitKey(1) == ord('q'):
            print("-- CV2 Stop --")
            break

    cv2.destroyAllWindows()
    _video_capture.release()
    get_logger().info(f"Video loader: {ts[0]/f:.6f} / Detector:{ts[1]/f:.6f} / Tracker: {ts[2]/f:.6f}")
