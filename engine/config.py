'''
Engine config. Hardcoded part.
'''

SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "gif", "original"]
SUPPORTED_PROFILES = ["price_detection"]
SUPPORTED_BACKENDS = ["TFSavedModelBackend", "TFServingRESTAPIBackend", "TFServingGRPCBackend"]
ENGINE_VERSION = 3.0
ENGINE_INFO = f"Engine v3.0\nSupports customer's profiles, model pool\n\tSupported image formats: {str(SUPPORTED_IMAGE_FORMATS)}\n\tSupported profiles: {str(SUPPORTED_PROFILES)}\n\tSupported model types:{str(SUPPORTED_BACKENDS)})."


# options for async  inference
MAX_GRPC_SESSIONS = 100
MAX_REST_SESSIONS = 100
MAX_LOCAL_SESSIONS = 10

