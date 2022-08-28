
''' 
Price tag parts/items for API 2.0 
Price always includes two parts/classes, the rest of items are named the same as a segment name in segmenation model's config
'''


# price tag items to recognize/extract and their classes
tag_items = {
                "BasePrice": ["iBasePrice", "fBasePrice"],
                "PromoPrice": ["iPromoPrice", "fPromoPrice"],
                "CardPrice": ["iCardPrice", "fCardPrice"],
                "OldPrice": ["iOldPrice", "fOldPrice"],
                "Name": [],
                "Description": [],                            
                "BarCode": [],
                "Vendor": [], 
                "Manufacturer": [], 
                "SKU": [], 
                "Date": [], 
                "Discount": [], 
                "Volume": [], 
                "Alc": [], 
                "HSPrice": ["iHSPrice", "fHSPrice"],     
                "ItemV": [], 
                "Country": [], 
                "QR": [], 
                "Category": [], 
                "BulkPrice": ["iBulkPrice", "fBulkPrice"], 
                "BulkAmount": [], 
                "Promo": [],                                        
                "address": []
        } 

DIGITAL_ITEMS = {"Discount", "Volume", "Alc", "ItemV" }

PRICE_ITEMS = {
        "BasePrice", 
        "PromoPrice", 
        "CardPrice", 
        "OldPrice", 
        "HSPrice", 
        "BulkPrice", 
        "iBasePrice",  "fBasePrice",
        "iPromoPrice", "fPromoPrice",
        "iCardPrice", "fCardPrice",
        "iOldPrice", "fOldPrice",
        "iHSPrice", "fHSPrice",
        "iBulkPrice", "fBulkPrice"

        }

TEXT_ITEMS = {"Name", "Description", "Vendor", "Manufacturer", "SKU", "Date", "Country", "Category", "address"}

GRAPHICAL_ITEMS = {"Promo"}