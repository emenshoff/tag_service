import os
#import cv2
# import pytesseract

# from engine.modules.ocr.ocr_backend import OCRBackend

# # Tessaract OCR lib parameters for the runtime
# tesseract_cfg = r"--oem 3 --psm 8"
# os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"


# class TessaractOCR(OCRBackend):
#     def __init__(self, config) -> None:
#         super().__init__(config)

#     def _preproicess_image(image):
#         # simple preprocessing
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (1, 1), 1.0)
#         _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         # thresh = ~thresh    
#         # thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
#         return thresh


#     def image_to_text(self, image) -> str:
#         preprocessed_img = self._preprocess_image(image)
#         pytesseract.image_to_string(preprocessed_img, lang='rus', config=tesseract_cfg)