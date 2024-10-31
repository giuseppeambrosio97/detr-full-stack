import base64
import binascii
from io import BytesIO
from PIL import Image, UnidentifiedImageError

class ImageDecodeError(Exception):
    """Custom exception for image decoding errors."""
    def __init__(self, message="Failed to decode base64 string into an image."):
        super().__init__(message)

class ImageEncodeError(Exception):
    """Custom exception for image encoding errors."""
    def __init__(self, message="Failed to encode image into a base64 string."):
        super().__init__(message)

def decode_image(image_base64: str) -> Image:
    """
    Decode a base64-encoded image string into a PIL Image.
    
    Args:
        image_base64 (str): Base64-encoded image string.
        
    Returns:
        Image: A PIL Image object.
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        return image
    except (binascii.Error, UnidentifiedImageError) as e:
        raise ImageDecodeError() from e

def encode_image(image: Image) -> str:
    """
    Encode a PIL Image into a base64 string.
    
    Args:
        image (Image): A PIL Image object.
        
    Returns:
        str: Base64-encoded image string.
    """
    try: 
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        # Encode the bytes to a base64 string
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return image_base64
    except Exception as e:
        raise ImageDecodeError() from e
