import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")


def transcribe_audio_file(audio_file_path: str) -> str:
    """
    Transcribe a WAV audio file using Azure Speech-to-Text.
    """

    if not SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY is missing from .env")

    if not SPEECH_REGION:
        raise ValueError("AZURE_SPEECH_REGION is missing from .env")

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(
        filename=audio_file_path
    )

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text

    elif result.reason == speechsdk.ResultReason.NoMatch:
        return ""

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails(result)
        raise Exception(
            f"Speech recognition canceled: {cancellation.reason}. "
            f"Error details: {cancellation.error_details}"
        )

    return ""