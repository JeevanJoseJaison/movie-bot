import os
import re
import html
import logging
import tempfile
import requests
from dotenv import load_dotenv
from flask import Flask, request, Response

from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity, ActivityTypes

from bot_logic import extract_preferences
from azure_openai_client import get_movie_recommendations
from speech_service import transcribe_audio_file
from vision_service import analyze_image_file


# --------------------------------------------------
# Setup
# --------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

load_dotenv()

MICROSOFT_APP_ID = os.getenv("MicrosoftAppId", "")
MICROSOFT_APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")

app = Flask(__name__)

adapter_settings = BotFrameworkAdapterSettings(
    MICROSOFT_APP_ID,
    MICROSOFT_APP_PASSWORD,
)

adapter = BotFrameworkAdapter(adapter_settings)


# --------------------------------------------------
# Formatting helpers
# --------------------------------------------------

def markdown_to_telegram_style(text: str) -> str:
    """
    Keeps text clean for Telegram through Azure Bot Service.
    Bot Framework channel rendering can differ, so this avoids broken formatting.
    """
    text = text.strip()

    # Remove excessive markdown if model returns it inconsistently
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", text)

    return text


def beautify_reply(reply: str) -> str:
    formatted = markdown_to_telegram_style(reply)

    return (
        "🎬 Your Movie Recommendations\n\n"
        f"{formatted}\n\n"
        "🍿 Enjoy your movie night!"
    )


def welcome_message() -> str:
    return (
        "🎬 Hi! I am your Movie Recommendation Bot\n\n"
        "You can send me:\n"
        "• Text — Recommend a funny sci-fi movie\n"
        "• Voice — Say what kind of movie you want\n"
        "• Image — Send a movie poster or scene image\n\n"
        "Examples:\n"
        "• Recommend a funny sci-fi movie\n"
        "• Suggest a family-friendly animated movie\n"
        "• I want something like Interstellar"
    )


# --------------------------------------------------
# Attachment helpers
# --------------------------------------------------

def download_attachment(content_url: str, file_suffix: str) -> str:
    """
    Downloads an attachment from Bot Framework contentUrl.
    Returns local temporary file path.
    """
    response = requests.get(content_url, timeout=60)

    if response.status_code != 200:
        raise Exception(
            f"Failed to download attachment: {response.status_code} - {response.text}"
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix)

    with open(temp_file.name, "wb") as file:
        file.write(response.content)

    return temp_file.name


def is_image_attachment(content_type: str) -> bool:
    return content_type and content_type.startswith("image/")


def is_audio_attachment(content_type: str) -> bool:
    return content_type and (
        content_type.startswith("audio/")
        or "ogg" in content_type
        or "mpeg" in content_type
        or "wav" in content_type
    )


# --------------------------------------------------
# Bot logic
# --------------------------------------------------

class MovieRecommendationBot:
    async def on_turn(self, turn_context: TurnContext):
        activity = turn_context.activity

        if activity.type != ActivityTypes.message:
            return

        user_text = activity.text or ""
        attachments = activity.attachments or []

        logger.info(f"Received activity text: {user_text}")
        logger.info(f"Received attachments: {len(attachments)}")

        try:
            # Start / welcome message
            if user_text.lower().strip() in ["/start", "start", "hi", "hello", "hey"]:
                await turn_context.send_activity(welcome_message())
                return

            # Handle image or voice attachment
            if attachments:
                await self.handle_attachment(turn_context, attachments, user_text)
                return

            # Handle normal text
            if user_text.strip():
                await self.handle_text(turn_context, user_text)
                return

            await turn_context.send_activity(
                "Please send a movie request, voice message, or image."
            )

        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            await turn_context.send_activity(
                "⚠️ Something went wrong while processing your request. Please try again."
            )

    async def handle_text(self, turn_context: TurnContext, user_text: str):
        logger.info(f"Processing text: {user_text}")
        print(f"[TEXT] {user_text}")

        prefs = extract_preferences(user_text)

        logger.info(f"Extracted preferences: {prefs}")
        print(f"[PREFS] {prefs}")

        reply = get_movie_recommendations(prefs)
        formatted_reply = beautify_reply(reply)

        await turn_context.send_activity(formatted_reply)

    async def handle_attachment(
        self,
        turn_context: TurnContext,
        attachments,
        user_text: str = "",
    ):
        attachment = attachments[0]

        content_type = attachment.content_type or ""
        content_url = attachment.content_url

        logger.info(f"Attachment content type: {content_type}")
        logger.info(f"Attachment content URL: {content_url}")

        if not content_url:
            await turn_context.send_activity(
                "I received an attachment, but I could not access its file URL."
            )
            return

        # Image handling
        if is_image_attachment(content_type):
            await self.handle_image(turn_context, content_url, user_text)
            return

        # Voice/audio handling
        if is_audio_attachment(content_type):
            await self.handle_audio(turn_context, content_url)
            return

        await turn_context.send_activity(
            f"I received an attachment, but this file type is not supported yet: {content_type}"
        )

    async def handle_image(
        self,
        turn_context: TurnContext,
        content_url: str,
        user_text: str = "",
    ):
        logger.info("Processing image attachment")
        print("[IMAGE] Processing image attachment")

        image_path = download_attachment(content_url, ".jpg")

        image_description = analyze_image_file(image_path)

        logger.info(f"Image analysis: {image_description}")
        print(f"[VISION] {image_description}")

        combined_text = (
            "Recommend movies based on this image description: "
            f"{image_description}. "
            f"Additional user text: {user_text}"
        )

        prefs = extract_preferences(combined_text)
        prefs["image_description"] = image_description

        logger.info(f"Image preferences: {prefs}")
        print(f"[IMAGE PREFS] {prefs}")

        reply = get_movie_recommendations(prefs)

        formatted_reply = (
            f"🖼️ Image analysis:\n{image_description}\n\n"
            f"{beautify_reply(reply)}"
        )

        await turn_context.send_activity(formatted_reply)

    async def handle_audio(self, turn_context: TurnContext, content_url: str):
        logger.info("Processing audio attachment")
        print("[VOICE] Processing audio attachment")

        audio_path = download_attachment(content_url, ".wav")

        transcript = transcribe_audio_file(audio_path)

        logger.info(f"Transcript: {transcript}")
        print(f"[TRANSCRIPT] {transcript}")

        if not transcript or transcript.strip() == "":
            await turn_context.send_activity(
                "🎙️ I could not clearly understand the voice message. "
                "Please try again with a short and clear voice note."
            )
            return

        prefs = extract_preferences(transcript)

        logger.info(f"Voice preferences: {prefs}")
        print(f"[VOICE PREFS] {prefs}")

        reply = get_movie_recommendations(prefs)

        formatted_reply = (
            f"🎙️ I heard: {transcript}\n\n"
            f"{beautify_reply(reply)}"
        )

        await turn_context.send_activity(formatted_reply)


bot = MovieRecommendationBot()


# --------------------------------------------------
# Flask routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return "Movie Recommendation Bot is running on Azure App Service."


@app.route("/api/messages", methods=["POST"])
def messages():
    if "application/json" not in request.headers.get("Content-Type", ""):
        return Response(status=415)

    body = request.json
    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    async def turn_handler(turn_context: TurnContext):
        await bot.on_turn(turn_context)

    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    task = adapter.process_activity(
        activity,
        auth_header,
        turn_handler,
    )

    loop.run_until_complete(task)

    return Response(status=201)


# --------------------------------------------------
# Run locally
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)