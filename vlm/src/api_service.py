import base64
from fastapi import FastAPI, Request

from VLMManager import VLMManager


app = FastAPI()

vlm_manager = VLMManager()


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/identify")
async def identify(instance: Request):
    """
    Performs Object Detection and Identification given an image frame and a text query.
    """
    # get base64 encoded string of image, convert back into bytes
    input_json = await instance.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        image_bytes = base64.b64decode(instance["b64"])

        bbox = vlm_manager.identify(image_bytes, instance["caption"])
        predictions.append(bbox)

    return {"predictions": predictions}


def unadjust_bbox(adjusted_bbox, original_size, target_size, pad_width, pad_height):
    """
    Revert the adjusted bounding box to its original dimensions.

    Args:
    adjusted_bbox (list): The bounding box adjusted for resized and padded image.
    original_size (tuple): Original dimensions of the image (width, height).
    target_size (int): The target size to which the longer side of the image was resized.
    pad_width (int): Padding added to the width to center the image.
    pad_height (int): Padding added to the height to center the image.

    Returns:
    list: The bounding box reverted to original image dimensions.
    """
    original_width, original_height = original_size
    scale_x = original_width / target_size
    scale_y = original_height / target_size

    # Unadjust bounding box
    x_min, y_min, x_max, y_max = adjusted_bbox
    x_min = (x_min - pad_width) * scale_x
    y_min = (y_min - pad_height) * scale_y
    x_max = (x_max - pad_width) * scale_x
    y_max = (y_max - pad_height) * scale_y
    original_bbox = [x_min, y_min, x_max, y_max]

    return original_bbox


def convert_bbox_to_dimensions(bbox): # CONVERT
    """
    Convert a bounding box from [x1, y1, x2, y2] to [x, y, width, height].

    Args:
    bbox (list or tuple): The bounding box in format [x1, y1, x2, y2].

    Returns:
    list: The bounding box in format [x, y, width, height].
    """
    x1, y1, x2, y2 = bbox
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return [x, y, width, height]