"""General scene text: region detection now, character recognition when available.

This is deliberately split in two, because the two halves have very different
dependency profiles:

1. **Text-region detection** - pure OpenCV (MSER + morphology). Finds *where*
   text-like structure is. Runs anywhere OpenCV runs, including this host.
2. **Character recognition** - needs a real OCR engine (tesseract, PaddleOCR,
   EasyOCR). None may be installed, and on a locked-down host none can be.

Splitting them means the useful half works today: Argus can report "there is
text here, at these coordinates, on this entity" and attach it to the scene
graph, then start reading the characters the moment an engine appears - with
no architectural change, because the region entities already exist.

**OCR is not LPR.** License-plate reading stays specialised: it has regional
formats, aspect-ratio priors and its own detector. This produces `TEXT`
entities that may belong to no other object at all - a street sign, a
storefront, a warning label.

Engines are probed by *use*, never assumed: `pytesseract` imports happily on a
machine with no `tesseract` binary and only fails when you call it. Anything
that fails is recorded as unavailable with the reason, and the region entities
are still produced.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .observation import BBox, Entity, EntityKind, Scene, Source

logger = logging.getLogger(__name__)

# Text regions smaller than this are unreadable at surveillance resolution;
# emitting them produces noise a reviewer has to wade through.
MIN_REGION_W = 22
MIN_REGION_H = 9

# Plausible aspect ratios for a run of text. Very tall or very wide blobs are
# structure (railings, window frames), not writing.
MIN_ASPECT = 1.1
MAX_ASPECT = 22.0

# A "text region" bigger than this share of the frame is the frame itself.
MAX_REGION_FRAME_SHARE = 0.5

# Characters an OCR engine may return that indicate noise rather than text.
_NOISE = re.compile(r"^[^\w]{1,}$")


class OcrEngine:
    """Wrapper around whichever OCR backend is actually usable.

    Availability is determined by *running* the engine once, not by importing
    it: `pytesseract` imports fine without the `tesseract` binary and only
    raises at call time. Import-only probing would advertise a capability that
    fails on the first real frame.
    """

    def __init__(self):
        self._backend: Optional[str] = None
        self._reason: str = "not probed"
        self._probed = False
        self._lock = threading.RLock()

    @property
    def backend(self) -> Optional[str]:
        self.probe()
        return self._backend

    @property
    def reason(self) -> str:
        self.probe()
        return self._reason

    @property
    def available(self) -> bool:
        return self.backend is not None

    def probe(self, force: bool = False) -> Optional[str]:
        with self._lock:
            if self._probed and not force:
                return self._backend
            self._probed = True
            self._backend = None

            # pytesseract: import AND execute, because the binary is separate.
            try:
                import numpy as np
                import pytesseract
                probe_img = np.full((32, 96, 3), 255, dtype=np.uint8)
                pytesseract.image_to_string(probe_img)
                self._backend = "pytesseract"
                self._reason = ""
                return self._backend
            except Exception as exc:  # noqa: BLE001
                self._reason = f"pytesseract unusable: {type(exc).__name__}"

            try:
                from paddleocr import PaddleOCR  # noqa: F401
                self._backend = "paddleocr"
                self._reason = ""
                return self._backend
            except Exception:
                pass

            try:
                import easyocr  # noqa: F401
                self._backend = "easyocr"
                self._reason = ""
                return self._backend
            except Exception:
                pass

            if not self._reason:
                self._reason = "no OCR engine installed"
            return None

    def read(self, crop) -> Optional[Tuple[str, float]]:
        """Read text from a crop. Returns (text, confidence) or None."""
        backend = self.backend
        if backend is None or crop is None:
            return None
        try:
            if backend == "pytesseract":
                import pytesseract
                # PSM 7 = a single line, which is what a cropped region is.
                text = pytesseract.image_to_string(
                    crop, config="--psm 7").strip()
                if not text or _NOISE.match(text):
                    return None
                # Tesseract's own confidence needs a second pass; the length
                # heuristic is honest about being a heuristic.
                return text, min(0.75, 0.35 + 0.05 * len(text))

            if backend == "paddleocr":
                from paddleocr import PaddleOCR
                engine = PaddleOCR(use_angle_cls=True, show_log=False)
                result = engine.ocr(crop, cls=True)
                if not result or not result[0]:
                    return None
                text, score = result[0][0][1]
                return str(text).strip(), float(score)

            if backend == "easyocr":
                import easyocr
                reader = easyocr.Reader(["en"], verbose=False)
                out = reader.readtext(crop)
                if not out:
                    return None
                _, text, score = out[0]
                return str(text).strip(), float(score)
        except Exception as exc:  # noqa: BLE001 - never break a frame
            logger.debug(f"OCR read failed: {exc}")
        return None


_ENGINE = OcrEngine()


def get_engine() -> OcrEngine:
    return _ENGINE


def _boxes_from_mask(mask, frame_area: float) -> List[BBox]:
    """Group a binary mask into plausible text lines."""
    import cv2

    # Wide, short kernel: joins glyphs horizontally into words and lines
    # without merging separate lines of text stacked vertically.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_REGION_W or h < MIN_REGION_H:
            continue
        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue
        # A region covering most of the frame is the frame, not a sign. This
        # is what a low-contrast or heavily textured scene degenerates into,
        # and passing it to an OCR engine would waste 120 ms on noise.
        if frame_area > 0 and (w * h) / frame_area > MAX_REGION_FRAME_SHARE:
            continue
        boxes.append(BBox(float(x), float(y), float(x + w), float(y + h)))
    return boxes


def _mser_mask(grey):
    """Glyph-shaped stable regions. Strong on real imagery, blind on flat
    synthetic images, which is why it is never used alone."""
    import cv2
    import numpy as np

    mser = cv2.MSER_create()
    try:
        mser.setMinArea(60)
        mser.setMaxArea(14000)
    except Exception:
        pass  # older OpenCV builds expose these only via the constructor

    regions, _ = mser.detectRegions(grey)
    mask = np.zeros(grey.shape, dtype=np.uint8)
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def _gradient_mask(grey):
    """High-gradient regions via a morphological gradient and Otsu.

    Polarity-independent: it responds to the *edges* of glyphs, so dark text
    on a light sign and illuminated text on a dark one are treated alike.
    MSER misses one of those cases depending on contrast direction.
    """
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _deduplicate(boxes: List[BBox], iou_threshold: float = 0.4) -> List[BBox]:
    """Drop boxes that substantially repeat one already kept.

    The two detectors legitimately find the same sign, and handing an OCR
    engine the same crop twice doubles the cost for nothing.
    """
    kept: List[BBox] = []
    for box in sorted(boxes, key=lambda b: -b.area):
        if any(box.iou(other) > iou_threshold for other in kept):
            continue
        kept.append(box)
    return kept


def find_text_regions(frame, max_regions: int = 12) -> List[BBox]:
    """Locate text-like regions without any OCR engine.

    Two complementary detectors are unioned because each has a blind spot:

    * **MSER** finds stable glyph blobs and works well on real, textured
      footage - but returns nothing on flat, low-noise images.
    * **Morphological gradient + Otsu** responds to glyph edges regardless of
      whether the text is lighter or darker than its background.

    Running both and merging costs ~2 ms and removes a whole class of
    "found nothing" failures that would otherwise look like "there is no text
    here" - a silent false negative, which is the failure mode this project
    treats as unacceptable.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return []
    if frame is None or getattr(frame, "size", 0) == 0:
        return []

    try:
        grey = frame if getattr(frame, "ndim", 0) == 2 else cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY)
        frame_area = float(grey.shape[0] * grey.shape[1])

        boxes: List[BBox] = []
        for build in (_gradient_mask, _mser_mask):
            try:
                boxes.extend(_boxes_from_mask(build(grey), frame_area))
            except Exception as exc:  # noqa: BLE001 - one detector failing is
                logger.debug(f"Text mask {build.__name__} failed: {exc}")

        # Largest first: the biggest text in frame is usually the informative
        # one (a sign rather than a bus timetable in the distance).
        boxes = _deduplicate(boxes)
        boxes.sort(key=lambda b: -b.area)
        return boxes[:max_regions]
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Text region detection failed: {exc}")
        return []


def extract_text(scene: Scene, frame, read: bool = True,
                 max_regions: int = 8) -> List[Entity]:
    """Add TEXT entities for every text-like region found in the frame.

    When an OCR engine is available the characters are attached as a `text`
    attribute; when not, the region is still recorded with
    `text_readable=False`, which is a genuine observation - Argus saw writing
    it could not read, and that is different from seeing no writing.
    """
    created: List[Entity] = []
    regions = find_text_regions(frame, max_regions=max_regions)
    if not regions:
        return created

    engine = get_engine()
    can_read = read and engine.available

    for box in regions:
        entity = Entity(kind=EntityKind.TEXT.value, category="text_region",
                        bbox=box, confidence=0.5, source=Source.OCR.value)
        entity.set_attribute("region_area_px", round(box.area, 1), 1.0,
                             Source.OCR.value)

        if can_read:
            try:
                x1, y1 = max(0, int(box.y1)), max(0, int(box.x1))
                crop = frame[int(box.y1):int(box.y2), int(box.x1):int(box.x2)]
                result = engine.read(crop)
            except Exception:
                result = None
            if result is not None:
                text, confidence = result
                entity.set_attribute("text", text, confidence, Source.OCR.value)
                entity.set_attribute("text_readable", True, 1.0, Source.OCR.value)
                entity.category = "text"
            else:
                entity.set_attribute("text_readable", False, 0.6,
                                     Source.OCR.value)
        else:
            entity.set_attribute("text_readable", False, 1.0, Source.OCR.value)
            entity.set_attribute("ocr_unavailable", engine.reason, 1.0,
                                 Source.OCR.value)

        scene.add_entity(entity)
        created.append(entity)
    return created


def attach_text_to_entities(scene: Scene, min_containment: float = 0.7) -> int:
    """Bind text regions to the object they sit on.

    Text on a van is a vehicle marking; text floating in space is signage.
    Distinguishing them is what makes the reading useful, so a contained
    region is recorded as an attribute of its parent as well as remaining an
    entity in its own right.
    """
    bound = 0
    text_entities = [e for e in scene.entities
                     if e.kind == EntityKind.TEXT.value and e.bbox is not None]
    others = [e for e in scene.entities
              if e.kind != EntityKind.TEXT.value and e.bbox is not None]

    for text_entity in text_entities:
        best, best_area = None, None
        for other in others:
            if other.bbox.contains(text_entity.bbox, min_fraction=min_containment):
                if best is None or other.bbox.area < best_area:
                    best, best_area = other, other.bbox.area
        if best is None:
            continue
        value = text_entity.get("text")
        if value:
            best.set_attribute("marking", value,
                               text_entity.get_attribute("text").confidence,
                               Source.OCR.value)
        else:
            best.set_attribute("has_unread_text", True, 0.6, Source.OCR.value)
        bound += 1
    return bound
