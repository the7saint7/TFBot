import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - Fusion

    {
        "name": "Allene (Fusion)",
        "avatar_path": "avatars/irene_allison.png",
        "folder": "allene",
        "message": " saw Allison and Irene high-five as they stepped on one of {BOT_NAME}'s magical trap. Soon their 3 bodies merge with them in full control of the resulting amalgam!"
    },

    {
        "name": "Joey (Fusion)",
        "avatar_path": "avatars/joey.png",
        "folder": "joey",
        "message": " dropped their notebook as John and Zoey crossed their path and accidentally activated on one of {BOT_NAME}'s magical trap. Soon their 3 bodies merge with them in full control of the resulting amalgam!"
    },

    {
        "name": "Jori (Fusion)",
        "avatar_path": "avatars/john_tori_fusion.png",
        "folder": "john_tori_fusion",
        "message": " picked up a strange ring. After putting it on, their bodies began to change. It was another prank by {BOT_NAME}. Quickly they become a sort of fusion of John and Tori Vegas."
    },

    {
        "name": "Jori2 (Fusion)",
        "avatar_path": "avatars/john_tori_headswap.png",
        "folder": "john_tori_headswap",
        "message": " accidentally sat on a magic trap! Soon, their bodies began to change. their bodies developed into the lustful form of Tori Vegas, but to everyone's shock, they have John's head!"
    },

    {
        "name": "Riyaka (Fusion)",
        "avatar_path": "avatars/riyaka.png",
        "folder": "riyaka",
        "message": " intervened in an argument between Sayaka and Rita and accidentally activated on one of {BOT_NAME}'s magical trap. Soon their 3 bodies merge with them in full control of the resulting amalgam!"
    },

    {
        "name": "Yuvia (Fusion)",
        "avatar_path": "avatars/flavia_yui.png",
        "folder": "flavia_yui",
        "message": " bumps into Flavia and Yui augmenting and accidentally trigger one of {BOT_NAME}'s trap. Soon their 3 bodies merge with them in full control of the resulting amalgam!"
    },

    {
        "name": "Zoel (Fusion)",
        "avatar_path": "avatars/flavia_yui.png",
        "folder": "zoel",
        "message": " saw Mel and Zoey high-five as they stepped on one of {BOT_NAME}'s magical trap. Soon their 3 bodies merge with them in full control of the resulting amalgam!"
    },



]