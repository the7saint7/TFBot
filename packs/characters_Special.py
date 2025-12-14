import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##Special

    {
        "name": "Narrator",
        "avatar_path": "",
        "folder": "narrator",
        "message": " found a microphone on the ground. A flash of light, a trapped glyph on the side and soon the air rippled like the start of a dream, and their body faded into sound. No hands. No face. Just a smooth, commanding voice echoing through the air. Every word they spoke bent reality a little, made people move, talk, blushâ€”just because they said so. They werenâ€™t part of the story anymore. They were the one telling it."
    },



]