import os
import requests
from dotenv import load_dotenv

load_dotenv()

VISION_KEY = os.getenv("AZURE_VISION_KEY")
VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")


def analyze_image_file(image_path: str) -> str:
    """
    Analyze a local image file using Azure AI Vision and return a short description.
    """

    if not VISION_KEY:
        raise ValueError("AZURE_VISION_KEY is missing from .env")

    if not VISION_ENDPOINT:
        raise ValueError("AZURE_VISION_ENDPOINT is missing from .env")

    url = (
        f"{VISION_ENDPOINT.rstrip('/')}"
        "/computervision/imageanalysis:analyze"
        "?api-version=2024-02-01&features=caption,tags"
    )

    headers = {
        "Ocp-Apim-Subscription-Key": VISION_KEY,
        "Content-Type": "application/octet-stream"
    }

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = requests.post(
        url,
        headers=headers,
        data=image_data,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"Azure Vision error: {response.status_code} - {response.text}")

    result = response.json()

    caption_text = ""
    tags_text = ""

    if "captionResult" in result:
        caption_text = result["captionResult"].get("text", "")

    if "tagsResult" in result:
        tags = result["tagsResult"].get("values", [])
        tag_names = [tag.get("name", "") for tag in tags if tag.get("name")]
        tags_text = ", ".join(tag_names[:10])

    description = f"Caption: {caption_text}. Tags: {tags_text}"

    return description