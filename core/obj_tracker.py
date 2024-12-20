import sys
from datetime import timedelta
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import get_logger
from core.tracking.person import Person


class ObjectTracker(object):
    """
    A generic object tracker class for integrating multiple tracking algorithms.

    Attributes:
        logger (Logger): Logger instance for debug and information logs.
        cfg (Config): Configuration object.
        tracker_type (str): Type of tracker used (e.g., 'sort').
        tracker (object): Initialized tracker instance.
    """
    def __init__(self, cfg):
        """
        Initializes the ObjectTracker class.

        Args:
            cfg (Config): Configuration object containing tracker settings.
        """
        self.logger = get_logger()
        self.cfg = cfg

        self.tracker_type = cfg.track_model_type.lower()

        if self.tracker_type == 'sort':
            from core.sort import Sort
            self.tracker = Sort(
                max_age=cfg.sort_max_age,
                min_hits=cfg.sort_min_hits,
                iou_threshold=cfg.sort_iou_thres,
            )

        self.persons = {}

        # logging
        self.f_cnt = 0
        self.t = 0

    def track(self, dets):
        """
        Tracks objects in the current frame using the initialized tracker.

        Args:
            dets (numpy.ndarray or list): Array of detections in the format [x1, y1, x2, y2, score, class].

        Returns:
            numpy.ndarray: Array of tracked objects in the format [x1, y1, x2, y2, tracking_id].
        """
        t0 = time.time()
        if self.tracker_type == 'sort':
            ret = self.tracker.update(dets)

        # Update or create Person instances
        for tracklet in ret:
            x1, y1, x2, y2, tracking_id = tracklet
            bbox = [x1, y1, x2, y2]
            person = self.persons.get(tracking_id) or Person(tracking_id)
            person.update(bbox)
            self.persons[tracking_id] = person

        # Clean up inactive persons
        removed_ids = Person.clean_up(timeout=timedelta(seconds=30))
        for removed_id in removed_ids:
            self.persons.pop(removed_id, None)

        t1 = time.time()

        # calculate time & logging
        self.f_cnt += 1
        self.t += t1 - t0
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.debug(
                f"{self.tracker_type} tracker {self.f_cnt} Frames average time - "
                f"track: {self.t / self.f_cnt:.6f} sec /")

        return ret

if __name__ == "__main__":
    import time
    import cv2

    from utils.config import set_config, get_config
    from utils.logger import init_logger
    from utils.visualize import get_color_for_id

    from core.obj_detector import ObjectDetector
    from core.media_loader import MediaLoader

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    media_loader = MediaLoader(_cfg.media_source,
                               logger=_logger,
                               realtime=_cfg.media_realtime,
                               bgr=_cfg.media_bgr,
                               opt=_cfg)

    _detector = ObjectDetector(cfg=_cfg)
    _tracker = ObjectTracker(cfg=_cfg)

    wt = 0 if media_loader.is_imgs else 1 / media_loader.dataset.fps

    while True:
        st = time.time()
        frame = media_loader.get_frame()

        _det = _detector.run(frame)

        for d in _det:
            _x1, _y1, _x2, _y2 = map(int, d[:4])
            cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)

        # tracking update
        if len(_det):
            track_ret = _tracker.track(_det)
            for t in track_ret:
                _x1, _y1, _x2, _y2 = map(int, t[:4])
                _tracking_id = int(t[4])
                color = get_color_for_id(_tracking_id)
                cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), color, thickness=2)
                cv2.putText(frame, f"ID: {_tracking_id}", (_x1, _y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

        et = time.time()
        if media_loader.is_imgs:
            delay = 0
        else:
            delay = int(max(1, (wt - (et - st)) * 1000)) if et - st < wt else 1

        cv2.imshow('_', frame)
        if cv2.waitKey(delay) == ord('q'):
            print("-- CV2 Stop --")
            break