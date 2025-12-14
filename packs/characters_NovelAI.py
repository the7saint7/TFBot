import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - NovelAI

    {
        "name": "StevieGB (NovelAI)",
        "folder": "stevieGB",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },



]