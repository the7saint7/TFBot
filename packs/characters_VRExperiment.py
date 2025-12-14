import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - VR Experiment

        {
        "name": "Hannah (VR Experiment)",
        "folder": "hannah",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "hikari (VR Experiment)",
        "folder": "hikari",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Hime (VR Experiment)",
        "folder": "hime",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Hina (VR Experiment)",
        "folder": "hina",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Hitomi (VR Experiment)",
        "folder": "hitomi",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "JaneBE (VR Experiment)",
        "folder": "jane_be",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "KatrinaBE (VR Experiment)",
        "folder": "katrina_be",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Meika (VR Experiment)",
        "folder": "meika",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Natsuki (VR Experiment)",
        "folder": "natsuki",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },

    {
        "name": "Reika (VR Experiment)",
        "folder": "reika_vrxp",
        "message": " finds one of {BOT_NAME}'s trap and transforms."
    },



]