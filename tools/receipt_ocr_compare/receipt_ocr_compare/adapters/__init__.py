from .base import BaseRecognizerAdapter
from .existing_project import ExistingProjectAdapter
from .paddleocr_recognizer import PaddleOCRRecognizerAdapter
from .svtrv2_b import SVTRv2BAdapter

__all__ = [
    "BaseRecognizerAdapter",
    "ExistingProjectAdapter",
    "PaddleOCRRecognizerAdapter",
    "SVTRv2BAdapter",
]

