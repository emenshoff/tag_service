'''
Price prediction service API data formats
'''

from pydantic import BaseModel
from typing import List, Optional
from engine.config import ENGINE_VERSION
from config import API_VERSION


# Request structure
class API_Request(BaseModel):    
    version: float
    profile : str
    content_type: str
    content_prefix: str        
    content: List[str]
    get_original_image: bool
    get_item_image: bool


# BarCode result
class API_BarCodeResult(BaseModel):
    items: List[dict] = None
    images: List[bytes] = None
    status: str = "success"
    err_msg: str = ""
    
# QR Code result
class API_QRCodeResult(BaseModel):
    items: List[str] = None
    images: List[bytes] = None
    status: str = "success"
    err_msg: str = ""

# Prediction fields for pricetag
class API_PriceTagFields(BaseModel):
    BasePrice: float = None
    PromoPrice: float = None
    CardPrice: float = None
    OldPrice: float = None
    Name: str = None
    Description: str = None
    BarCode: API_BarCodeResult = None
    Vendor: str = None
    Manufacturer: str = None
    SKU: str = None
    Date: str = None
    Discount: float = None
    Volume: float = None
    Alc: float = None
    HSPrice: float = None
    ItemV: float = None
    Country: str = None
    QR: API_QRCodeResult = None
    Category: str = None
    BulkPrice: float = None
    BulkAmount: float = None
    Promo: bool = None
    address: str = None

# Prediction item
class API_PriceTagPrediction(BaseModel):
    tagclass: str = "Unknown"
    brand: str = "Unknown"
    promo: bool = False
    pricetag_image: bytes = None
    items: API_PriceTagFields


# Image processing result
class API_ResultItem(BaseModel):
    original_image: bytes = None
    result_status: str = "success"
    error_message: str = ""
    prediction: API_PriceTagPrediction = None


# Response structure
class API_Response(BaseModel):
    version: float = API_VERSION
    engine_verion: float = ENGINE_VERSION
    status: str = "success"
    error_message: str = ""
    content: List[API_ResultItem]


class ProtocolError(Exception):
    pass
