"""Stage 3b: open-vocabulary detection on candidate events only.

A closed-set detector cannot name a thing it was never trained on, and an
anomaly is almost by definition exactly that. YOLO-World takes a text prompt
list as its class list at inference time - no training required. Cheap
because, like the VLM, it only ever looks at the ~8 candidate windows
geometry already flagged, never the full 18,000 frames.
"""
import cv2


class NoopOpenVocab:
    def detect(self, video_path, ev, prompts):
        return {"hits": [], "reason": "open_vocab.backend=none - not run"}


class YoloWorldOpenVocab:
    def __init__(self, cfg):
        from ultralytics import YOLO
        ov = cfg["open_vocab"]
        self.cfg = ov
        self.model = YOLO(ov["weights"])
        self.prompts = ov["prompts"]
        self.model.set_classes(self.prompts)

    def detect(self, video_path, ev, prompts=None):
        """Runs the text-prompted detector on a few frames inside the event
        window. Returns boxes for classes the closed-set detector has no
        label for at all."""
        prompts = prompts or self.prompts
        if prompts != self.prompts:
            self.model.set_classes(prompts)
            self.prompts = prompts

        k = self.cfg["frames_per_event"]
        cap = cv2.VideoCapture(video_path)
        times = [ev.t_start + (ev.t_end - ev.t_start) * i / max(1, k - 1)
                 for i in range(k)]
        hits = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            r = self.model.predict(frame, conf=self.cfg["conf"], verbose=False)[0]
            b = r.boxes
            if b is None:
                continue
            for xyxy, cf, cls_idx in zip(b.xyxy.cpu().tolist(), b.conf.cpu().tolist(),
                                          b.cls.int().cpu().tolist()):
                hits.append({"t": round(t, 2), "prompt": self.prompts[cls_idx],
                            "conf": round(float(cf), 3), "xyxy": [round(v, 1) for v in xyxy]})
        cap.release()
        return {"hits": hits, "reason": None}


def build_open_vocab(cfg):
    backend = cfg.get("open_vocab", {}).get("backend", "none")
    if backend == "none":
        return NoopOpenVocab()
    if backend == "yoloworld":
        return YoloWorldOpenVocab(cfg)
    raise ValueError("unknown open_vocab backend: " + backend)
