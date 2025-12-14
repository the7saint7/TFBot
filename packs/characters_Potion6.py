import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - Potion #6

    {
        "name": "JohnGB (Potion #6)",
        "avatar_path": "avatars/johnGB_ptn6.png",
        "folder": "johnGBPt6",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "JohnGB - EmoChick (Potion #6)",
        "avatar_path": "avatars/emoChick.png",
        "folder": "emoChick",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "KiyoshiGB (Potion #6)",
        "avatar_path": "avatars/kiyoshiGB_ptn6.png",
        "folder": "kiyoshiGBPtn6",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "ZoeyGB (Potion #6)",
        "avatar_path": "avatars/zoeyGB.png",
        "folder": "zoeyGB",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },



]