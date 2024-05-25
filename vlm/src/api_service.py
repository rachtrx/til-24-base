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