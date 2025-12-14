import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - Shorts

    {
        "name": "CorneliaGB (Shorts)",
        "avatar_path": "avatars/corneliaGB.png",
        "folder": "corneliaGB",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "Hinano (Shorts)",
        "avatar_path": "avatars/hinano.png",
        "folder": "hinano",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "JackGB (Shorts)",
        "avatar_path": "avatars/jackGB.png",
        "folder": "jackGB",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "John OneEye (Shorts)",
        "avatar_path": "avatars/john_oneEye.png",
        "folder": "johnOneEye",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "John Styled (Shorts)",
        "avatar_path": "avatars/john_styled.png",
        "folder": "johnStyled",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "KatrinaBimbo (Shorts)",
        "avatar_path": "avatars/katrinaBimbo.png",
        "folder": "katrinaBimbo",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "Kutsumi (Shorts)",
        "avatar_path": "avatars/kutsumi.png",
        "folder": "kutsumi",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "Reika (Shorts)",
        "avatar_path": "avatars/reika.png",
        "folder": "reika",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },

    {
        "name": "Sera (Shorts)",
        "avatar_path": "avatars/sera.png",
        "folder": "serena",
        "message": " finds one of {BOT_NAME}'s trap and becomes transforms."
    },



]