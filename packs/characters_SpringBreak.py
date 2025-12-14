import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - SpringBreak

    {
        "name": "HollyBimbo (SpringBreak)",
        "avatar_path": "avatars/hollyBimbo.png",
        "folder": "hollyBimbo",
        "message": "dabs a gloss from a tube {BOT_NAME} 'enchanted,' and a fizzy warmth blooms behind their smile. Thoughts go bubbly, body language turns luxe and inviting, and suddenly everything feels like a spotlight flirting back."
    },

    {
        "name": "JanePunk (SpringBreak)",
        "avatar_path": "avatars/janePunk.png",
        "folder": "janePunk",
        "message": "buckles a studded cuff that hums with {BOT_NAME}'s mischief, and a bassline thrums under their skin. Their smirk sharpens, rules feel optional, and a delicious urge to start trouble (and finish it) takes the lead."
    },

    {
        "name": "Nikki (SpringBreak)",
        "avatar_path": "avatars/nikki.png",
        "folder": "nikki",
        "message": "tightens a hand-wrap inked with {BOT_NAME}'s rune, and heat spikes up their arms. Their stance squares, muscles wake, and a reckless, competitive grin settles in like they were born to throw the first punch and win the last word."
    },

    {
        "name": "Ririsa (SpringBreak)",
        "avatar_path": "avatars/ririsa.webp",
        "folder": "ririsa",
        "message": "snaps a stubborn hairclip that {BOT_NAME} primed, and a tiny spark runs along their scalp. Lines go crisp, posture precise, and their voice sharpens into cool, organized biteâ€”half sass, half 'don't waste my time.'"
    },

    {
        "name": "Youko (SpringBreak)",
        "avatar_path": "avatars/youko.webp",
        "folder": "youko",
        "message": "thumbs the rhinestone phone charm {BOT_NAME} hexed, and a warm buzz pours through them. Hair falls glossy, jewelry seems to gleam hotter, and their mood tilts queen-beeâ€”casual, flirty, and absolutely sure the room will orbit."
    },

    {
        "name": "YuiBimbo (SpringBreak)",
        "avatar_path": "avatars/yuiBimbo.png",
        "folder": "yuiBimbo",
        "message": "pops open a compact with a pink kiss-rune, and a sweet shimmer rolls down their cheeks. Their giggle hangs bright, their pose goes playful, and they canâ€™t help batting lashes like teasing is a love language."
    },



]