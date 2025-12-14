import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##Student Transfer

    {
        "name": "Abby Luten",
        "avatar_path": "avatars/abby.png",
        "folder": "abby",
        "message": "walks through the office door and straight into a quiet shimmer that {BOT_NAME} left hanging there. Their outfit tightens into something formal, their posture firms, and the urge to take responsibility for everyone hits like a headache waiting to happen."
    },

    {
        "name": "Alex Hawkins",
        "avatar_path": "avatars/alex.png",
        "folder": "alex",
        "message": "picks up a coachâ€™s whistle that wasnâ€™t there a second ago, and the sharp little flash that follows leaves them standing solid and confident in gym-wear. Their mind clicks into health-class mode, already ready to lecture and push limits."
    },

    {
        "name": "Allison Stein",
        "avatar_path": "avatars/allison.png",
        "folder": "allison",
        "message": "brushes a cute little heart sticker {BOT_NAME} left on a locker and feels a slow, floaty warmth spill through them. Their look softens, their expression goes dreamy, and their thoughts start drifting somewhere gentle and a little spaced out."
    },

    {
        "name": "Amy",
        "avatar_path": "avatars/amy.png",
        "folder": "amy",
        "message": "trips over a glowing charm {BOT_NAME} left in the hallway, and a rush of warmth climbs their neck. Their look turns tidy, their thoughts go organized, and suddenly they're managing everyone's chaos like it's a full-time job."
    },

    {
        "name": "Anuja Killada",
        "avatar_path": "avatars/anuja.png",
        "folder": "anuja",
        "message": "grabs a mug that {BOT_NAME} left behind, and the scent alone sends a hum through their skin. Their posture eases, their eyes narrow with quiet amusement, and their mind starts solving problems before anyone finishes speaking."
    },

    {
        "name": "Audrey Callahan",
        "avatar_path": "avatars/audrey.png",
        "folder": "audrey",
        "message": "leans into a pink-glow charm {BOT_NAME} 'forgot' on the couch, and a mellow warmth spreads through them until their outfit hangs cute and casual. Their mood slides easy and social, like it's already a party and they're already the draw."
    },

    {
        "name": "Brad Chambers",
        "avatar_path": "avatars/brad.png",
        "folder": "brad",
        "message": "grabs a gym towel that's quietly pulsing with {BOT_NAME}'s magic, and a warm jolt runs through them. Their stance settles into easy jock confidence, and they suddenly feel stupidly protective of the people around them."
    },

    {
        "name": "BradGB",
        "avatar_path": "avatars/bradgb.png",
        "folder": "bradGB",
        "message": "brushes against a locker door marked with {BOT_NAME}'s seal, and a pulse of magic bends their reflection into a confident smirk. Their stride smooths out, their tone sharpens, and they start seeing every hallway as their runway."
    },

    {
        "name": "Carla",
        "avatar_path": "avatars/carla.png",
        "folder": "carla",
        "message": "touches a glowing whistle near the gym bench, and a buzz of authority tightens through their stance. Their voice picks up that competitive edge, and they suddenly know how to motivate or terrify anyone on command."
    },

    {
        "name": "Carrie Parker",
        "avatar_path": "avatars/carrie.png",
        "folder": "carrie",
        "message": "finds a strange mirror note that {BOT_NAME} left behind and can't resist reading it. A warm pulse rolls through them, leaving a sharp, magnetic confidence behindâ€”Carrie's charm sticking like perfume."
    },

    {
        "name": "Cassie Wright",
        "avatar_path": "avatars/cassie.png",
        "folder": "cassie",
        "message": "checks her reflection in a compact that's humming with {BOT_NAME}'s magic, and in one blink their look sharpens into money and expectation. They feel that practiced confidence settle in, the kind that assumes service and doesn't wait in lines."
    },

    {
        "name": "Catrina",
        "avatar_path": "avatars/catrina.png",
        "folder": "catrina",
        "message": "triggers one of {BOT_NAME}'s weirder pranksâ€”a charm meant for 'feline agility training.' A puff of magic later, their balance is flawless, ears twitch at every sound, and they can't shake the urge to chase anything that moves."
    },

    {
        "name": "Charlotte Foster",
        "avatar_path": "avatars/charlotte.png",
        "folder": "charlotte",
        "message": "opens a sketchbook with weird little runes in the margins, and color bleeds up their arms like wet paint. Clothes tilt playful, energy turns bright, and their head starts racing with ideas they suddenly need to show everyone right now."
    },

    {
        "name": "Circe",
        "avatar_path": "avatars/circe.png",
        "folder": "circe",
        "message": " bet with Circe they could do a better job granting wish and so Circe obliged, swapping their lives. For a time..."
    },

    {
        "name": "Claus Hawkins",
        "avatar_path": "avatars/claus.png",
        "folder": "claus",
        "message": "touches a Student Council armband thatâ€™s been marked with a tiny rune, and the air tightens around them. Their posture goes strict and efficient, and their thoughts swing straight to rules, order, and taking charge."
    },

    {
        "name": "Connie Williams",
        "avatar_path": "avatars/connie.png",
        "folder": "connie",
        "message": "clicks a red grading pen that hums in their fingers, and their clothes pull tidy and professional all at once. Their voice drops into a calm teacher register before they even speak, and patience settles in like it belongs there."
    },

    {
        "name": "Cornelia Roberts",
        "avatar_path": "avatars/cornelia.png",
        "folder": "cornelia",
        "message": "steps on a sparkle sticker {BOT_NAME} left on the cheer mat, and a bright jolt pops through their body, bouncing their stance lighter. All at once they just want to stick close, smile big, and make sure the right person is impressed."
    },

    {
        "name": "Dominic Carter",
        "avatar_path": "avatars/dominic.png",
        "folder": "dominic",
        "message": "adjusts a server apron from Serenade and doesnâ€™t notice the faint shimmer stitched into it. Their expression smooths into polite charm, and they slip into that calm, practiced tone of someone living off tips and patience."
    },

    {
        "name": "Donna Barnes",
        "avatar_path": "avatars/donna.png",
        "folder": "donna",
        "message": "brushes a stack of â€˜urgentâ€™ forms and sets off a neat little glow. Their outfit pulls sharp and office-clean, their face settles into quiet control, and their brain starts sorting problems like everyone else is already late."
    },

    {
        "name": "Elizabeth Sebas",
        "avatar_path": "avatars/elizabeth.png",
        "folder": "elizabeth",
        "message": "touches a silver crest {BOT_NAME} shouldn't even have, and a soft wash of light smooths every movement they make. Their clothes refine into proper service, and a calm sense of duty slides in like they were born to take care of someone important."
    },

    {
        "name": "Emily Brooks",
        "avatar_path": "avatars/emily.png",
        "folder": "emily",
        "message": "kicks through a burst of confetti-looking magic under the table and canâ€™t help perking up. Their look shifts into bold, casual energy, and their brain instantly wants to try something dumb, new, and fun in that exact order."
    },

    {
        "name": "Eric Tyner",
        "avatar_path": "avatars/eric.png",
        "folder": "eric",
        "message": "lights up a cigarette that flickers purple instead of amber, and a lazy wave rolls through them. Their body language goes slouched and cocky, and their thoughts drift into that troublemaker mindset that never takes school seriously."
    },

    {
        "name": "EricGB",
        "avatar_path": "avatars/ericgb.png",
        "folder": "ericGB",
        "message": " touched the silver crest {BOT_NAME} shouldn't even have left behind, and the world ripples around them. Their body softens, curves blooming where none were before, their hair grows long as their clothes shift to fit their new body. A teasing echo of {BOT_NAME}'s laughter lingers in their head as a wave of calm obedience settles in. Somehow, serving feelsâ€¦ rightâ€”like this was the prank all along."
    },

    {
        "name": "Faith Reinhardt",
        "avatar_path": "avatars/faith.png",
        "folder": "faith",
        "message": "touches a half-faded charm stuck in the locker room tiles and feels a cold ripple pass straight through them. Their edges feel lighter, voice softer, and thereâ€™s this strange, lonely echo in their thoughts that doesnâ€™t feel like theirs."
    },

    {
        "name": "Flavia Lucca",
        "avatar_path": "avatars/flavia.png",
        "folder": "flavia",
        "message": "grabs a swim cap thatâ€™s quietly glowing poolside, and a smooth, athletic confidence rolls over them like water. Their balance shifts into that easy captainâ€™s poise, and they suddenly feel used to being watched and cheered for."
    },

    {
        "name": "Frogyaka",
        "avatar_path": "avatars/sayaka_frog.png",
        "folder": "sayaka_frog",
        "message": " steps on green paste and a flash of magic surrounds them. A sudden chill runs through their veins as their skin turns slick and green. Their arms and legs shrink, folding tight beneath them, and the world around them grows impossibly huge. Their clothes vanish in a puff of magic, replaced by a single startled croak. {BOT_NAME}'s laughter echoes somewhere aboveâ€”'Ribbit for me, cutie.'"
    },

    {
        "name": "Gabriel Riel",
        "avatar_path": "avatars/gabriel.png",
        "folder": "gabriel",
        "message": "opens a weird little charm covered in symbols and hears it hum back. A cool stillness settles over them, and they start looking at everyone like a walking experiment that needs to be documented and judged."
    },

    {
        "name": "Genny Collins",
        "avatar_path": "avatars/genny.png",
        "folder": "genny",
        "message": "opens a study guide that {BOT_NAME} 'helped' annotate, and a cool little pulse slips in behind their eyes. Their face settles calm, their posture relaxed, and their thoughts turn clever in a way that feels a little dangerous to everyone else."
    },

    {
        "name": "Grace Reinhardt",
        "avatar_path": "avatars/grace.png",
        "folder": "grace",
        "message": "leans against the infirmary desk and sets off a faint glow in the paperwork pile. Their expression sinks into that â€˜Iâ€™ve seen worseâ€™ look, and their first instinct is to tell everyone to sit down and stop being dramatic."
    },

    {
        "name": "Greg Fisk",
        "avatar_path": "avatars/greg.png",
        "folder": "greg",
        "message": "picks up a bio textbook that crackles with {BOT_NAME}'s mark, and a sharp clarity hits fast. Their tone shifts to firm lecture mode, and they feel this urge to demand focus and reward 'good effort' like a real teacher."
    },

    {
        "name": "Gwen Strauss",
        "avatar_path": "avatars/gwen.png",
        "folder": "gwen",
        "message": "tries on a bracelet from the rack, not noticing the tiny rune worked into the tag, and a soft flash runs along their arms. Their look turns sharp retail-cute, and their patience drops to â€˜customer service minimumâ€™ right away."
    },

    {
        "name": "Hank Vega",
        "avatar_path": "avatars/hank.png",
        "folder": "hank",
        "message": "leans on a set of enchanted car keys {BOT_NAME} left on the counter, and a grounded weight settles in. They feel steady, stubborn, a little lonely, and way too ready to act like they're responsible for everybody's safety."
    },

    {
        "name": "Holly Davis",
        "avatar_path": "avatars/holly.png",
        "folder": "holly",
        "message": "opens a message pinged from an unknown contact (yeah, {BOT_NAME}), and a low hum curls tight in their chest. Their expression cools, their guard slides up, and their thoughts pull inward like they're done letting anyone see too much."
    },

    {
        "name": "Irene Virelles",
        "avatar_path": "avatars/irene.png",
        "folder": "irene",
        "message": "laces up a spare pair of court shoes with a faint mark burned into the sole, and a hard, steady focus settles through their whole frame. Protective instinct spikes fast, and theyâ€™re suddenly ready to get between someone and trouble."
    },

    {
        "name": "Izuna Ito",
        "avatar_path": "avatars/izuna.png",
        "folder": "izuna",
        "message": "rests a hand on a practice sword with runes along the hilt, and the air tightens. Their stance centers, their movements go controlled and efficient, and their thoughts drop into quiet discipline with no room for nonsense."
    },

    {
        "name": "Jack Mallory",
        "avatar_path": "avatars/jack.png",
        "folder": "jack",
        "message": "brushes a history binder that sparks at his touch, and the air seems to get meaner. Their voice turns sharp and smug, and they get that urge to lecture about pain and consequences just to watch people squirm."
    },

    {
        "name": "John Davis",
        "avatar_path": "avatars/john.png",
        "folder": "john",
        "message": "picks up a notebook that {BOT_NAME} absolutely wasn't supposed to have access to, and a soft glow crawls up their arm. Their thoughts start sliding into quick improvisation and damage control, like chaos is normal and they're built for it."
    },

    {
        "name": "JohnGB",
        "avatar_path": "avatars/johngb.png",
        "folder": "johnGB",
        "message": "steps right into one of {BOT_NAME}'s trapsâ€”a flash of violet light ripples through them, and their grin tilts sly as a new kind of mischief bubbles under the surface. They blink, confused, but the voice that escapes is smoother than before."
    },

    {
        "name": "Johngb3",
        "avatar_path": "avatars/johngb3.png",
        "folder": "johnGB3",
        "message": " stepped on a trap left by {BOT_NAME} and were transported in an alternate universe where they became a different version of John."
    },

    {
        "name": "Katrina Morgan",
        "avatar_path": "avatars/katrina.png",
        "folder": "katrina",
        "message": "pokes a glowing doodle on a locker without thinking, and a rush of heat runs up their spine until theyâ€™re loose in the shoulders and already grinning. That playful, â€˜letâ€™s stir it a littleâ€™ mood settles in before they can stop it."
    },

    {
        "name": "KatrinaGB",
        "avatar_path": "avatars/katrinagb.png",
        "folder": "katrinaGB",
        "message": "picks up a notebook that {BOT_NAME} left behind, and a soft glow crawls up their arm. Soon they are transformed into a handsome young man filled with curiosity and a desire to experiment with their new perspective."
    },

    {
        "name": "Kiyo",
        "avatar_path": "avatars/kiyoshigb.png",
        "folder": "kiyoshiGB",
        "message": "hesitate after hearing a familiar giggle. But too late they feel their hair grow very long and their frame grow smaller. Soon they have a small pair of breast of their own..."
    },

    {
        "name": "Kiyoshi Honda",
        "avatar_path": "avatars/kiyoshi.png",
        "folder": "kiyoshi",
        "message": "grabs a USB stick labeled â€˜classifiedâ€™ in marker, and the thing actually hums. A rush of paranoid excitement fires through them, and suddenly every hallway shadow feels like proof of the conspiracy theyâ€™ve been talking about for years."
    },

    {
        "name": "Kyoko",
        "avatar_path": "avatars/kyoko.png",
        "folder": "kyoko",
        "message": "picks up a book with a faint purple mark, and light spills between the pages. Calm settles in instantly, their shoulders relax, and they find themselves quietly smiling like they already know what happens next."
    },

    {
        "name": "Laura Morgan",
        "avatar_path": "avatars/laura.png",
        "folder": "laura",
        "message": "touches a glowing post-it on the fridge that definitely wasnâ€™t there five minutes ago. Their clothes pull into office-clean lines, their shoulders pick up a tired weight, and their mind starts trying to keep track of everyone at once."
    },

    {
        "name": "Leona Winters",
        "avatar_path": "avatars/leona.png",
        "folder": "leona",
        "message": "picks up a teacup from the garden table and catches the soft shimmer clinging to it. A gentle calm smooths their face, their voice warms, and a little flicker of loneliness sneaks in under the polite smile."
    },

    {
        "name": "Maria Grandales",
        "avatar_path": "avatars/maria.png",
        "folder": "maria",
        "message": "lifts a camera charm {BOT_NAME} clipped to the strap, and a bright flash hits them before the shutter even clicks. Their stance angles into 'looking for a story,' and their thoughts start hunting for dirt like it's oxygen."
    },

    {
        "name": "Marty",
        "avatar_path": "avatars/marty.png",
        "folder": "marty",
        "message": "wipes down a bar glass that {BOT_NAME} quietly tagged, and a smooth warmth spreads through them. Their smile turns easy, their voice drops to playful, and they start reading people like regulars waiting to spill secrets."
    },

    {
        "name": "Matt Davidson",
        "avatar_path": "avatars/matt.png",
        "folder": "matt",
        "message": "picks up a retro game case with a faint glow at the seam, and a nostalgic buzz hits. Their mood turns laid-back and slightly flirty, and they start thinking about settling in, getting comfortable, and not being alone tonight."
    },

    {
        "name": "Maurice Honda",
        "avatar_path": "avatars/maurice.png",
        "folder": "maurice",
        "message": "taps a police-style notepad {BOT_NAME} spelled over, and a low, observant calm moves in. They start tracking details in the room automatically, carrying themselves like the tired adult who's seen it all and still shows up anyway."
    },

    {
        "name": "Mel Larner",
        "avatar_path": "avatars/mel.png",
        "folder": "mel",
        "message": "leans back against a glowing vending machine, and the hum slides straight into their bones. Their look softens, their mood turns flirtatious, and suddenly every word they say feels like an invitation to trouble."
    },

    {
        "name": "Michelle Bloom",
        "avatar_path": "avatars/michelle.png",
        "folder": "michelle",
        "message": "picks up a pair of glasses with a faint rune along the frame, and a quick spark settles behind their eyes. Their face cools into that smart, planning look, and theyâ€™re already thinking three moves ahead like itâ€™s natural."
    },

    {
        "name": "Mina Hubbard",
        "avatar_path": "avatars/mina.png",
        "folder": "mina",
        "message": "touches a charm bracelet sitting on the counter, and the little glow that follows makes them draw in on themself. Their presence softens, their shoulders tuck in, and a quiet need to stay close to someone safe drifts in with it."
    },

    {
        "name": "Naomi Nakano",
        "avatar_path": "avatars/naomi.png",
        "folder": "naomi",
        "message": "leans back against a jacket {BOT_NAME} warded and feels a slow, confident warmth slide over them. Their body language settles into protective and flirty at the same time, already playing guard and bad influence in one breath."
    },

    {
        "name": "Nemuri Otani",
        "avatar_path": "avatars/nemuri.png",
        "folder": "nemuri",
        "message": "rests a hand on a shrine charm thatâ€™s been tampered with, and a quiet weight wraps around them. They feel protective and worn, with that deep, constant worry that comes from taking care of someone who needs you."
    },

    {
        "name": "Nora Whittaker",
        "avatar_path": "avatars/nora.png",
        "folder": "nora",
        "message": "picks up a stack of lesson plans that hum quietly in her hands, and a soft, steady calm spills through them. Their tone turns reassuring, their presence warm, and they suddenly feel responsible for keeping everyone okay."
    },

    {
        "name": "Patty",
        "avatar_path": "avatars/patty.png",
        "folder": "patty",
        "message": "slides a finger across a tarot card with a glowing edge, and a slow grin forms before they realize it. Their movements go smooth and suggestive, and their thoughts pick up that smug little â€˜I already know how this endsâ€™ vibe."
    },

    {
        "name": "Paul Simmons",
        "avatar_path": "avatars/paul.png",
        "folder": "paul",
        "message": "opens a glowing planner covered in tiny notes, and a warm little rush lands behind their eyes. They slide right into friendly nerd mode, half apologetic and half proud, and already ready to overshare lore nobody asked for."
    },

    {
        "name": "Phila Hubbard",
        "avatar_path": "avatars/phila.png",
        "folder": "phila",
        "message": "drops down on a couch cushion with a faint sigil burned into the fabric, and a firm, controlling weight settles into their posture. Their stare sharpens, and suddenly theyâ€™re ready to run the room and defend whoever needs it."
    },

    {
        "name": "Rachel Clark",
        "avatar_path": "avatars/rachel.png",
        "folder": "rachel",
        "message": "taps a blue-lit detention slip that {BOT_NAME} definitely forged, and the air tightens around them. Their outfit shifts casual but authoritative, and their thoughts lock into that cool teacher mode that expects obedience right now."
    },

    {
        "name": "Riley Evans",
        "avatar_path": "avatars/riley.png",
        "folder": "riley",
        "message": "picks up a tiny folded note with {BOT_NAME}'s sigil on it, and a nervous spark jumps in their chest. Their stance goes a little smaller, their face goes red fast, and they can't stop thinking about how cute everybody suddenly looks."
    },

    {
        "name": "RileyGB",
        "avatar_path": "avatars/rileygb.png",
        "folder": "rileyGB",
        "message": "drinks a strange soda with a magical curse on it. Next thing they know, they are weak and pathetic Riley. Forced to dress 'en femme' by their best friend Genny and come to the party."
    },

    {
        "name": "Rita Piper",
        "avatar_path": "avatars/rita.png",
        "folder": "rita",
        "message": "hits a rune-etched wrap in the boxing room and feels a hot spike of attitude ride up their spine. Their stance squares up fast, and their mouth is suddenly loaded with the kind of blunt trash talk that starts fights on purpose."
    },

    {
        "name": "Rudy",
        "avatar_path": "avatars/rudy.png",
        "folder": "rudy",
        "message": "sniffs a strange glowing tennis ball {BOT_NAME} left in the yard, and the air pops with magic. A quick shimmer runs through them, tail wag slowing as awareness clicks onâ€”they're still loyal, still curious, but now definitely understand every sarcastic thing humans say."
    },

    {
        "name": "Sadie-Lynn Kobayashi",
        "avatar_path": "avatars/sadie.png",
        "folder": "sadie",
        "message": "grabs a spare cheer bow thatâ€™s glowing just a little too bright and feels a pop of energy jump through them. They light up instantly, hungry for attention and approval, already posing like theyâ€™re new and out to prove it."
    },

    {
        "name": "Sandra Davis",
        "avatar_path": "avatars/sandra.png",
        "folder": "sandra",
        "message": "steps on a sigil {BOT_NAME} hid in the kitchen tile and feels a warm, teasing buzz circle their chest and throat. Their smile turns knowing, their tone turns mom-sweet and nosy, and suddenly they're already judging someone's love life."
    },

    {
        "name": "Sayaka Sato",
        "avatar_path": "avatars/sayaka.png",
        "folder": "sayaka",
        "message": "touches a glitter-marked cheer ribbon and a warm pulse runs through them, leaving them standing like they own the hallway. Hair falls just right, smile turns practiced sweet, and suddenly everyone else feels a level below them."
    },

    {
        "name": "Scarlet Morgan",
        "avatar_path": "avatars/scarlet.png",
        "folder": "scarlet",
        "message": "takes a sip from a coffee cup that {BOT_NAME} shouldn't have been able to enchant, and a slow, sleepy sigh rolls through them. Their shoulders slump into college-student exhaustion, and their brain snaps straight to 'I've got work due.'"
    },

    {
        "name": "Serena",
        "avatar_path": "avatars/serena.png",
        "folder": "serena",
        "message": "picked up a small fluffy ball when suddenly a flash of light blinds them. It was another prank sigil left by {BOT_NAME}. Their ears stretch high, soft fur sprouting across their arms and legs as a fluffy tail pops into place. When the light fades, they're left standing as a startled rabbit-girlâ€”long-eared, wide-eyed, and very, very naked. {BOT_NAME}'s giggle in the distance makes it clear the missing clothes weren't an accident."
    },

    {
        "name": "Setsuna Otani",
        "avatar_path": "avatars/setsuna.png",
        "folder": "setsuna",
        "message": "touches a paper charm from the shrine and feels a quiet, gentle pull settle through them. Their movements get careful, their voice softens, and that lonely patience creeps in like waiting became a way of life."
    },

    {
        "name": "Syn",
        "avatar_path": "avatars/syn.png",
        "folder": "syn",
        "message": " bumps into {BOT_NAME} and switches bodies with them!"
    },

    {
        "name": "Tori Vega",
        "avatar_path": "avatars/tori.png",
        "folder": "tori",
        "message": "grabs a jacket with a glowing stitch along the sleeve and feels her pulse kick hard. Their stance drops into ready-to-swing, and their thoughts turn sharp and defensive, already daring anyone to start something."
    },

    {
        "name": "Vanessa Russell",
        "avatar_path": "avatars/vanessa.png",
        "folder": "vanessa",
        "message": "touches a scuffed hallway locker that hums under their palm, and a jittery rush hits them like theyâ€™re caught mid-scheme. They feel bold, reckless, and one bad idea away from following Tori into trouble without thinking."
    },

    {
        "name": "Yui Yamashita",
        "avatar_path": "avatars/yui.png",
        "folder": "yui",
        "message": "sets off a tiny council seal {BOT_NAME} stuck to a rulebook, and their clothes snap into a crisp uniform while their stance straightens on instinct. Their thoughts narrow fast toward order, schedules, and fixing everyone else's behavior."
    },

    {
        "name": "Yuuna Yamashita",
        "avatar_path": "avatars/yuuna.png",
        "folder": "yuuna",
        "message": "picks up a half-folded bill with a tiny rune in the corner and feels the weight land immediately. Their shoulders tense with tired responsibility, and all they can think about is keeping everyone else okay, no matter what it costs."
    },

    {
        "name": "Zoey Chambers",
        "avatar_path": "avatars/zoey.png",
        "folder": "zoey",
        "message": "touches a strange phone device that {BOT_NAME} left behind, and a jolt of pink light dances up their arm. Their grin widens, their body language loosens, and that unstoppable influencer energy takes over like it owns the moment."
    },



]