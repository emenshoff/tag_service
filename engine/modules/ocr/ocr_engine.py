"""(Tesseract-based)"""

import logging

# from engine.modules. import TessaractOCR

# from engine.modules.ocr.pytessaract_ocr import TessaractOCR

SUPPORTED_OCR_BACKENDS = ["tessaract"]

DEFAULT_OCR_BACKEND = "tessaract"


log = logging.getLogger('engine')


class OCR:
    def __init__(self, backend) -> None:
        try:
            if backend in SUPPORTED_OCR_BACKENDS:
                if backend == "tesseract":
                    # self._backend = TessaractOCR()
                    raise Exception(f'Not supported OCR backend: {backend}')
                else:
                    raise Exception(f'Not supported OCR backend: {backend}')
            else: 
                raise Exception(f'Not supported OCR backend: {backend}')
        except Exception as ex:
            err_msg = f"OCR backend error: {ex}"
            log.error(err_msg)
            raise Exception(err_msg)
  

    def recognize(self, image) -> str:
        return self._backend.image_to_text(image)
       