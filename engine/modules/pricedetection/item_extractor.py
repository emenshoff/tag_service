'''
Price tag items extractor

'''
from re import I
import base64
#import cv2

import numpy as np

from pyzbar import pyzbar

from config import OCR_ENABLED, OCR_BACKEND

from engine.modules.ocr.ocr_engine import OCR
from .digit_detection import DigitDetector
from .tag_items import tag_items, DIGITAL_ITEMS, TEXT_ITEMS, GRAPHICAL_ITEMS, PRICE_ITEMS
from api.protocol import API_PriceTagFields, API_PriceTagPrediction, API_QRCodeResult, API_ResultItem, API_BarCodeResult


from utils.io import encode_jpeg
# from utils.image_preprocessing import sharpen_filter, brightness_contrast_filter


class TagItemExtractor:
    ''' Extracts / Interprets tag items after tag semantic recognition '''
    def __init__(self, config, model_pool):
        self._digit_detector = DigitDetector(model_pool[config["digit_detector"]])
        self._string_classes = ["Name"]
        self._attach_img_classes = ["BarCode", "QR"]
    
    
    # ObjectDetection - based OCR for digits
    async def _extract_digit(self, img):
        result = await self._digit_detector.process([img])
        return result[0]


    def _extract_text(self, img):
        ''' OCR functionality  '''
        text_extractor = OCR(OCR_BACKEND)
        recognized_text = text_extractor.recognize(img)
        return recognized_text


    def _extract_barcode(self, img):
        ''' Bar Code extraction. uses pyzbar + zbar lib '''
                
        result = []
        try:
            # preprocessing_fn = brightness_contrast_filter # sharpen_filter(img)
            # preprocessed_img = preprocessing_fn(img)
            # img = sharpen_filter(img)
            decoded_objects = pyzbar.decode(img)
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    continue
                (x, y, w, h) = obj.rect
                rect = (y, x, h, w)
                result.append(
                    (
                        obj.type,
                        obj.data.decode('utf-8'),
                        rect
                    )
                )
            return result
        except Exception as ex:
            msg = f'barcode recognition failed, error: {ex}'
            raise Exception(msg)


    def _extract_qr_code(self, img):
        ''' QR code extractor '''
        result = []
        try:
            
            # img = sharpen_filter(img)
            decoded_objects = pyzbar.decode(img)
            for obj in decoded_objects:
                if obj.type != 'QRCODE':
                    continue
                (x, y, w, h) = obj.rect
                rect = (y, x, h, w)
                result.append(
                    (
                        obj.type,
                        obj.data.decode('utf-8'),
                        rect
                    )
                )
            return result
        except Exception as ex:
            msg = f'QR recognition failed, error: {ex}'
            raise Exception(msg)
            

    async def process(self, tag_img, segments, get_item_image=False):
        ''' 
        exctracts semantic items from single image segments dict 
            img: tag image
            segments: tag segments extracted my semantic recognition model
            get_item_image: attach segment's images to the result
            returns dict {
                segment's_name: [segment's recognized_items]
            }
        '''

        def check_img_shape(img):
            ''' checking the image before feed CNN '''            
            try:
                h = img.shape[0] 
                w = img.shape[1]
                c = img.shape[2]
            except Exception as ex:
                # error_msg = f"wrong image format! exception={ex}"
                return False

            if h <= 0 or w <= 0 or c != 3:
                # error_msg =  f"wrong image_shape! shape = {img.shape}"
                return False
            
            return True

       
        # print(f"Segments: {segments}")
        result = {key: None for key in tag_items.keys()}  # if class / segment is not been found on the image let's mark it with None value 

               
        for tag_item in tag_items:            # cycle over all tags
            
            # checking price items         
            if tag_item in PRICE_ITEMS:    # price has two classes                
                i, f = None, None             # integer and fractional parts of a price
                for seg_name in tag_items[tag_item]:                    
                    segment_items = segments.get(seg_name, None)   # REFACTOR FOR MULTIPLE SEGMENT IMAGES!!!         
                    #--- insert           
                    if segment_items is not None:
                        # taking only one price segment item
                        img = segment_items[0][-1]
                        # cv2.imshow(f'seg_name', img)
                        # cv2.waitKey()
                        digit_val = await self._extract_digit(img)
                        # ДОБАВИТЬ СКЛЕЙКУ СЕГМЕНТОВ
                        if seg_name.startswith("i"):
                            i = digit_val                        
                        elif seg_name.startswith("f"):
                            if digit_val / 10 < 1:
                                digit_val /= 10
                            else: 
                                digit_val /= 100                         

                # else result is None (not detected)                
                value = None
                # checking calue parts
                if i is None: #                     
                    if f is None:  # 
                        value = None
                    else:
                        value = f   # only fraction detected
                else:
                    if f is None:
                        value = i   # only integer detected
                    else:
                        value = i + f                    
                
                result[tag_item] = value
               

            # checking digital items         
            elif tag_item in DIGITAL_ITEMS:
                segment_items = segments.get(tag_item, None)
                if segment_items is not None:                              
                    img = segment_items[0][-1] 
                    # cv2.imshow(f'segment {tag_item}', img)
                    # cv2.waitKey()
                    if check_img_shape(img):
                        digit_val = await self._extract_digit(img)
                    else:
                        digit_val = 0.
                    result[tag_item] = digit_val


            # qr code
            elif tag_item == "QR":
                segment_items = segments.get(tag_item, None)
                if segment_items is not None:     
                    status = "failue"            
                    err_msg = ""    

                    result[tag_item] = dict()
                    result[tag_item]["items"] = []
                    result[tag_item]["images"] = []               
                    
                    
                    for _, _, _, img in segment_items:
                        if img is not None:                            
                            if get_item_image:                                                   # attach segment's images if exists                   
                                if check_img_shape(img):                             
                                    encoded_jpeg = encode_jpeg(img).numpy()
                                    result_img = base64.b64encode(encoded_jpeg).decode('utf-8')
                                    result[tag_item]["images"].append(result_img)  
                    try:
                        qr_data = self._extract_qr_code(tag_img)
                        if len(qr_data) > 0:
                            status = "success"
                            for qr_item in qr_data:
                                result[tag_item]["items"].append(qr_item)

                        elif len(qr_data) > 1:
                            err_msg = f"More than one QR code found: {len(qr_data)}"
                        else:
                            err_msg = "QR code is not readable."
                    except Exception as ex:
                        err_msg = f'QR code was not recognized, recognition library fail: {ex}'
                    
                    result[tag_item]["status"] = status
                    result[tag_item]["err_msg"] = err_msg
                     
                    

            # bar code. using original tag image, not segment image, because it works better
            elif tag_item == "BarCode":                
                segment_items = segments.get(tag_item, None)              
                if segment_items is not None:
                    status = "failue"            
                    err_msg = ""    

                    result[tag_item] = dict()
                    result[tag_item]["items"] = []
                    result[tag_item]["images"] = []

                    # attach segment's images if exists
                    if get_item_image:
                        for _, _, _, img in segment_items:
                            if img is not None:
                                if check_img_shape(img):                                   
                                    encoded_jpeg = encode_jpeg(img).numpy()
                                    result_img = base64.b64encode(encoded_jpeg).decode('utf-8')
                                    result[tag_item]["images"].append(result_img)                 
                    try:
                        bar_codes = self._extract_barcode(tag_img)
                        if len(bar_codes) > 0:
                            status = "success"
                            for i, (_type, _value, _) in enumerate(bar_codes):
                                result[tag_item]["items"].append(
                                    {
                                        "type": _type,                      # bar code type: EAN13, CODE128 e.t.c.
                                        "value": _value                     # bar code value
                                    }
                                )

                        elif len(bar_codes) > 1:
                            err_msg = f"More than one bar code found: {len(bar_codes)}"
                        else:
                            err_msg = "Bar code is not readable."
                    except Exception as ex:
                        err_msg = f'bar_code not recognized, recognition library fail: {ex}'
                    
                    result[tag_item]["status"] = status
                    result[tag_item]["err_msg"] = err_msg
                     

            elif tag_item in TEXT_ITEMS:
                segment_items = segments.get(tag_item, None)
                if segment_items is not None:
                    # print(f"Found segment image for {tag_item} segment")                     
                    if tag_item in self._attach_img_classes:
                        str_result = ""
                        for _, _, _, img in segment_items:
                            if img is not None:
                                if check_img_shape(img):                                                     
                                    encoded_jpeg = encode_jpeg(img).numpy()
                                    result_img = base64.b64encode(encoded_jpeg).decode('utf-8')
                                    if result[tag_item] is None:
                                        result[tag_item] = list()
                                    if tag_item in self._string_classes:
                                        if OCR_ENABLED:                             # Tessaract OCR critically (x2) slows down image processing, so let's make it optional
                                            str_result += self._extract_text(img)
                                        result[tag_item].append(result_img)
                                    else:
                                        result[tag_item].append(result_img)
                        if tag_item in self._string_classes and str_result != "":
                            result[tag_item].append(str_result)
                        # else:
                        #     result_img = base64.b64encode(encoded_jpeg).decode('utf-8')
                        #     result[tag_item] = result_img
            
            elif tag_item in GRAPHICAL_ITEMS:
                if get_item_image:  
                    if tag_item in self._attach_img_classes:                  
                        segment_items = segments.get(tag_item, None)
                        if segment_items is not None:
                            result[tag_item] = []
                            for _, _, _, img in segment_items:
                                if img is not None:
                                    if check_img_shape(img):                                                    
                                        encoded_jpeg = encode_jpeg(img).numpy()
                                        result_img = base64.b64encode(encoded_jpeg).decode('utf-8')                                        
                                        result[tag_item].append(result_img)
                        
            else:
                result[tag_item] = None



        # assign two price values : BasePrice and PromoPrice    
        '''
        Логика определения цен
        1) Необходимо выделять две цены: базовая и промо
        2) Если на ценнике одна цена - то она считается базовой
        3) Если на ценнике две цены:
            старая и промо: старая (зачеркнутая) считается базовой
            базовая и по карте: по карте считается промо
            базовая, оптовая и по карте: оптовую не считаем, по карте считается промо
            базовая и по банковской карте: по банковской карте считается промо
        '''
        # base_price = 0.0
        # promo_price = 0.0
        prices = {key: value for key, value in result.items() if key in PRICE_ITEMS}
        non_zero_prices = {key: value for key, value in prices.items() if (value is not None) and (value != 0.0)}
        base_price_subs = ["OldPrice"]        
        promo_prices_subs = ["CardPrice"]#, "HSPrice", "BulkPrice"]

        for price_name, price_value in non_zero_prices.items():
            # base price. take the maximum one
            if price_name in base_price_subs:
                if result["BasePrice"] == None or result["BasePrice"] == 0.0:
                    result["BasePrice"] = price_value               
                else:
                    result["BasePrice"] = max(result["BasePrice"], price_value)
            # promo price. take the minimum one
            elif  price_name in promo_prices_subs:
                if result["PromoPrice"] == None or result["PromoPrice"] == 0.0:
                    result["PromoPrice"] = price_value
                else:
                    result["PromoPrice"] = min(result["PromoPrice"], price_value)


        # the only price is BasePrice always
        if result["BasePrice"] == None or result["BasePrice"] == 0.0:
            if result["PromoPrice"] != None and result["PromoPrice"] != 0.0:
                result["BasePrice"] = result["PromoPrice"]
                result["PromoPrice"] = None

        # костыль для КБ. (исправить дообучением моделей)
        if result["PromoPrice"] != None:
            if result["BasePrice"] != None:
                if result["PromoPrice"] / 1 < 1:    # no integer part
                    if result["BasePrice"] % 1 == 0: # no float part
                        result["BasePrice"] += result["PromoPrice"]
                    result["PromoPrice"] = None
                    

        return API_PriceTagFields(**result)
