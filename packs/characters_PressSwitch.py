import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##Press Switch

    {
        "name": "Alma Beoulve (Press Switch)",
        "avatar_path": "avatars/alma.png",
        "folder": "PS_alma",
        "message": "picks up {BOT_NAME}'s remote, and reality snaps into place. Their posture straightens with quiet authority, and a faint Japanese accent colors their words. Their thoughts narrow toward order and simplicity, and they find themselves moving with the weight of someone who's lived far from home."
    },

    {
        "name": "Amber Jinsen (Press Switch)",
        "avatar_path": "avatars/amber.png",
        "folder": "PS_amber",
        "message": "picks up {BOT_NAME}'s remote, and the world flickers like a broken screen. When it stabilizes, they're standing impossibly tall, their body a patchwork of mismatched parts. Their mind feels simple and dopey, but their eyes light up at handsome faces and beautiful forms like they're seeing the world through someone else's desires."
    },

    {
        "name": "Anna Greenfield (Press Switch)",
        "avatar_path": "avatars/anna.png",
        "folder": "PS_anna",
        "message": "touches a device {BOT_NAME} left behind, and the air around them warps like heat shimmer. Their body shifts into curves that demand attention, their voice dropping into a southern drawl they've never had. Suddenly they're flirting with everyone in sight, moving like youth and desire are everything."
    },

    {
        "name": "AnnaAR (Press Switch)",
        "avatar_path": "avatars/anna.png",
        "folder": "PS_annaAR",
        "message": "picks up {BOT_NAME}'s remote, and time seems to reverse around them. Their skin smooths, their body tightens, and they feel decades younger. The southern charm comes easier now, as if this body remembers a life they never lived, and every movement feels lighter and more natural."
    },

    {
        "name": "Aprika (Press Switch)",
        "folder": "PS_aprika",
        "message": "touches a device {BOT_NAME} left behind, and two realities merge into one. Their body becomes a strange amalgam, their thoughts a confusing mix of judgmental directness and aloof curiosity. They're not quite either person, but something newâ€”a glitch that {BOT_NAME} created just to see what would happen."
    },

    {
        "name": "April Cox (Press Switch)",
        "avatar_path": "avatars/april.png",
        "folder": "PS_april",
        "message": "picks up {BOT_NAME}'s remote, and reality glitches hard. Their accent shifts to Australian, their posture turns judgmental and direct. They're already sizing everyone up like they're grading effort, and honesty cuts sharp where excuses don't fly."
    },

    {
        "name": "AprilGB (Press Switch)",
        "folder": "PS_aprilGB",
        "message": "picks up a device {BOT_NAME} left behind, and the world flips like a coin. Their body changes, their voice deepens, but that judgmental streak stays sharp as ever. They're still sizing people up and calling out mistakes, but something feels slightly off about everything around them."
    },

    {
        "name": "Arelia Lorisa Olivian (Press Switch)",
        "folder": "PS_arelia",
        "message": "picks up {BOT_NAME}'s remote, and their voice smooths into a proper British accent they've never had. Their posture turns elegant, their words eloquent, and they move like they've been acting on soap operas their whole life. Real life and fiction blur together, where performance and truth are the same."
    },

    {
        "name": "Ashley Degard (Press Switch)",
        "folder": "PS_ashley",
        "message": "touches a device {BOT_NAME} left behind, and their voice gains an angelic quality. Their posture straightens with moral authority, and they're suddenly judging everyone's reputation while trying not to be boring. Trauma hides behind a perfect facadeâ€”they're the beautiful angel everyone sees, even as they struggle inside."
    },

    {
        "name": "AshleyGB (Press Switch)",
        "folder": "PS_ashleyGB",
        "message": "picks up a device {BOT_NAME} left behind, and reality flips sideways. Their body changes, their voice deepens, but that moral judgment stays sharp. They're still trying not to be boring, still judging everyone's reputation, but everything feels slightly off."
    },

    {
        "name": "Aurora Petrice (Press Switch)",
        "folder": "PS_aurora",
        "message": "picks up a device {BOT_NAME} left behind, and their voice shifts into a valley girl accent they've never had. Their posture turns vain and know-it-all, their thoughts focused on psychology and being the smartest in the room, even as their appearance screams bimbo. Intelligence and appearance don't matchâ€”they're both everything and nothing like they seem."
    },

    {
        "name": "Betty Valentine (Press Switch)",
        "folder": "PS_betty",
        "message": "touches a device {BOT_NAME} left behind, and their voice drops into crass, expletive-laden speech they've never used. Their posture turns tomboy-tough, their words shorten into 'fuckin' this' and 'fuckin' that,' and they're already sizing up the room like a natural contrarian. Femininity feels like a weakness here, and being direct is everythingâ€”they're the smartass with a comment for everything."
    },

    {
        "name": "Calvin Hintre (Press Switch)",
        "folder": "PS_main",
        "message": "picks up {BOT_NAME}'s remote, and their body changes as their perspective flips. Suddenly they're seeing the world through eyes that have lived a different life entirely, and they're the main character of someone else's story. Their choices matter in ways they never expectedâ€”they're both observer and participant in something that isn't their own."
    },

    {
        "name": "Candice Hoover (Press Switch)",
        "folder": "PS_candice",
        "message": "touches a device {BOT_NAME} left behind, and reality warps around them like a funhouse mirror. Their body shifts into something controlled and calculating, their thoughts turning to experiments and manipulation. They're the one pulling strings nowâ€”the puppet master, not the puppet."
    },

    {
        "name": "Cassandra Brooks (Press Switch)",
        "folder": "PS_cassandra",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture turns confident and friendly, their thoughts focused on police work and loyalty. They find themselves in a body that's been solving cases, and they're Julie's friend and partnerâ€”law enforcement and friendship intertwined."
    },

    {
        "name": "Chastity Degard (Press Switch)",
        "folder": "PS_chastity",
        "message": "touches a device {BOT_NAME} left behind, and reality glitches hard. Their body shifts into something new, their thoughts turning to motherhood and responsibility. They find themselves as Ashley's overworked motherâ€”exhaustion and love are the same thing, keeping everything together even as they fall apart."
    },

    {
        "name": "Christine Olivian Jr (Press Switch)",
        "folder": "PS_chris",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. Their posture turns proper and controlled, and they find themselves as the daughter of the Olivian household. Elegance and control are everythingâ€”their family name carries weight, and their choices matter in ways they never expected."
    },

    {
        "name": "Ciel Mariana Pire (Press Switch)",
        "folder": "PS_ciel",
        "message": "touches a device {BOT_NAME} left behind, and reality warps like a broken screen. Their body shifts into a maid's uniform, their thoughts turning to service and submission. They're the one serving, not being servedâ€”their identity tied to someone else's needs, where being a maid is both curse and purpose."
    },

    {
        "name": "Cindy Fable (Press Switch)",
        "folder": "PS_cindy",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture turns playful and mischievous, their thoughts focused on nightlife and friends. They're the one causing trouble with Jenna and Reinaâ€”the life of the party where fun and chaos are the same thing."
    },

    {
        "name": "Dilbert Frowny (Press Switch)",
        "folder": "PS_dilbert",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. Their expression turns perpetually frowny, and they find themselves with a strange name and stranger appearance. Being different is normal hereâ€”their frown is just part of who they are."
    },

    {
        "name": "Donald Barron (Press Switch)",
        "folder": "PS_donald",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. Their posture turns teacher-authoritative, and they find themselves teaching at Eliza's school. Education and authority are intertwinedâ€”they're the one shaping minds, not being shaped."
    },

    {
        "name": "DonaldGB (Press Switch)",
        "folder": "PS_donaldGB",
        "message": "picks up a device {BOT_NAME} left behind, and reality flips sideways. Their body changes, their voice shifts, but that teacher authority stays sharp. Still the one in charge, still shaping mindsâ€”just from a different angle in a world where everything's slightly off."
    },

    {
        "name": "Eliza Hintre (Press Switch)",
        "folder": "PS_eliza",
        "message": "touches a device {BOT_NAME} left behind, and their body shifts. Their voice gains a playful edge, and they find themselves as Calvin's sister. Sibling rivalry and love are the same thingâ€”they're the one causing trouble, not watching it happen."
    },

    {
        "name": "Erin Beoulve (Press Switch)",
        "folder": "PS_erin",
        "message": "walks through a rift that {BOT_NAME} tore open, and their body shifts into a tomboy form. Their thoughts turn to action and leadership, and they're the one taking chargeâ€”being bold and direct is everything, leaping into action instead of waiting for permission."
    },

    {
        "name": "Ermach (Press Switch)",
        "folder": "PS_ermach",
        "message": "touches a device {BOT_NAME} left behind, and three realities merge into one chaotic form. Their body becomes a strange amalgam of Erin, Martha, and Christine, their thoughts a confusing mix of action, responsibility, and control. Not quite any single person, but something newâ€”a glitch where three lives have merged into one impossible existence."
    },

    {
        "name": "Fair Wallow (Press Switch)",
        "folder": "PS_fair",
        "message": "steps through a glitch that {BOT_NAME} created, and their body shifts. Their posture turns judgmental and sharp, their thoughts focused on reputation and consequences. They're the one gathering crowds to ruin peopleâ€”being loud and angry is a weapon, causing scenes instead of avoiding them."
    },

    {
        "name": "Fat Woman (Press Switch)",
        "folder": "PS_fwoman",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes into a larger form. Their presence becomes more noticeable, and their size is just part of who they areâ€”being different is normal here, just another person in a world that doesn't judge."
    },

    {
        "name": "Heather Hintre (Press Switch)",
        "folder": "PS_mother",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. Their posture turns maternal and controlling, and they find themselves as Calvin's mother. Motherhood and control are intertwinedâ€”they're the one running the household, not following orders."
    },

    {
        "name": "Hillary Mounse (Press Switch)",
        "folder": "PS_hillary",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts. Their thoughts turn to loyalty and following, and they find themselves as one of Erin's followers. Being part of a group is everythingâ€”supporting the leader instead of leading."
    },

    {
        "name": "Hoover (Press Switch)",
        "folder": "PS_hoover",
        "message": "touches a device {BOT_NAME} left behind, and their body shifts into an alternate form of Candice. Their thoughts turn to experiments and control, and they're the one running the showâ€”an alternate version of someone else, a different name but the same purpose."
    },

    {
        "name": "Hope Neesli (Press Switch)",
        "folder": "PS_hope",
        "message": "walks through a portal that {BOT_NAME} opened, and their body changes. Their posture turns gentle and uncertain, and they find themselves as someone who likes karaoke but is scared to sing. Fear and desire are intertwinedâ€”they want to be bold but can't quite manage it."
    },

    {
        "name": "Howard O'reily (Press Switch)",
        "folder": "PS_howard",
        "message": "touches a device {BOT_NAME} left behind, and their body shifts. Their thoughts turn to observation and patience, and they're the one watching and waiting. Being steady and reliable is everythingâ€”they're the one others depend on."
    },

    {
        "name": "Iida Hilard (Press Switch)",
        "folder": "PS_iida",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. Their posture turns proper and controlled, and they find themselves with a Japanese name and proper demeanor. Tradition and modernity clashâ€”they're caught between two worlds."
    },

    {
        "name": "Jaina (Press Switch)",
        "folder": "PS_jaina",
        "message": "touches a device {BOT_NAME} left behind, and their body shifts. Their thoughts turn to observation and understanding, and they're the one seeing things others don't. Perception is everythingâ€”they're the one with the unique perspective."
    },

    {
        "name": "Jenna Hintre (Press Switch)",
        "folder": "PS_jenna",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts. Their posture turns playful and social, their thoughts focused on music and friends. They find themselves as Calvin's other sisterâ€”the peacemaker where being the middle child means keeping everyone together."
    },

    {
        "name": "Jennifer Fidget (Press Switch)",
        "folder": "PS_jennifer",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. Their movements become fidgety and nervous, and they find themselves as one of Erin's loyal followers. Loyalty and anxiety are the same thingâ€”they're always on edge, always ready to follow."
    },

    {
        "name": "Jillian Maxwell (Press Switch)",
        "folder": "PS_jillian",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. Their posture turns professional and efficient, and they find themselves as Christine's housekeeper. Service and dignity are intertwinedâ€”they're the one keeping things running, even as they remain invisible."
    },

    {
        "name": "Julie Lighte (Press Switch)",
        "folder": "PS_julie",
        "message": "picks up a device {BOT_NAME} left behind, and reality glitches. Their body shifts into something new, their thoughts focused on police work and justice, and they find themselves as the one solving cases. Law enforcement and friendship are the same thingâ€”they're the one protecting people, not the one being protected."
    },

    {
        "name": "Karyn Cooper (Press Switch)",
        "folder": "PS_karyn",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on observation and judgment, and they find themselves as the one watching and waiting. Being patient and observant is everythingâ€”they're the one who sees everything but says nothing."
    },

    {
        "name": "Kayla Orion (Press Switch)",
        "folder": "PS_kayla",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. Their presence becomes faceless and undefined, and they find themselves as one of Erin's drones. Being part of the crowd is everythingâ€”they're the one who follows whoever is popular, not the one who leads."
    },

    {
        "name": "Kenichi (Press Switch)",
        "folder": "PS_kenichi",
        "message": "touches a device {BOT_NAME} left behind, and reality glitches. Their body shifts into something new, their thoughts focused on tradition and honor, and they find themselves as the one with the Japanese name and proper values.  culture and identity are everythingâ€”they're caught between old ways and new."
    },

    {
        "name": "Martha Vilsong (Press Switch)",
        "folder": "PS_martha",
        "message": "walks through a portal that {BOT_NAME} opened, and their body changes. Their posture turns responsible and tired, and they find themselves as the one carrying the weight. Responsibility and exhaustion are the same thingâ€”they're the one everyone depends on, even as they're falling apart."
    },

    {
        "name": "Megan Curtain (Press Switch)",
        "folder": "PS_megan",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on friendship and support, and they find themselves as the one helping April and Megumi. Being a good friend is everythingâ€”they're the one people turn to when they need help."
    },

    {
        "name": "MeganTan (Press Switch)",
        "folder": "PS_megan_tan",
        "message": "picks up {BOT_NAME}'s remote, and their body changes where they're the tanned version of themselves. Their body changes, their skin darkens, but that friendly supportive nature stays the same. They're still the one helping others, still the good friendâ€”just from a different angle in a world where everything's slightly different."
    },

    {
        "name": "Megumi Nalashibara (Press Switch)",
        "folder": "PS_megumi",
        "message": "picks up a device {BOT_NAME} left behind, and reality snaps. Their body changes, their posture turns proper and Japanese, and they find themselves as they're the one with the traditional name and modern struggles.  culture and identity clashâ€” they're caught between two worlds, trying to find their place."
    },

    {
        "name": "MegumiTan (Press Switch)",
        "folder": "PS_megumi_tan",
        "message": "picks up {BOT_NAME}'s remote, and their body changes where they're the tanned version of themselves. Their body changes, their skin darkens, but that struggle between tradition and modernity stays the same. They're still caught between two worldsâ€”just from a different angle in a reality where everything's slightly off."
    },

    {
        "name": "Melina Cooper (Press Switch)",
        "folder": "PS_melina",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns friendly and approachable, and they find themselves as they're the one people turn to for advice. Being supportive and understanding is everythingâ€” they're the one who listens, not the one who talks."
    },

    {
        "name": "Melody Geneway (Press Switch)",
        "folder": "PS_melody",
        "message": "picks up a device {BOT_NAME} left behind, and reality glitches. Their body shifts into something new, their thoughts focused on music and rhythm, and they find themselves as the one creating harmony.  sound and emotion are the same thingâ€”they're the one who brings people together through music."
    },

    {
        "name": "Michael Geneway (Press Switch)",
        "folder": "PS_mike",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on responsibility and family, and they find themselves as Melody's brother. Being the older sibling means being the protectorâ€”they're the one looking out for others, not the one being looked after."
    },

    {
        "name": "MichelleCloisewater (Press Switch)",
        "folder": "PS_michelle",
        "message": "picks up a device {BOT_NAME} left behind, and reality snaps. Their body changes, their posture turns intelligent and planning, and they find themselves as they're the one thinking three moves ahead.  strategy and control are everythingâ€” they're the one manipulating situations, not the one being manipulated."
    },

    {
        "name": "Mika Greenfield (Press Switch)",
        "folder": "PS_mika",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns aloof and curious, and they find themselves as they're Anna and Sean's adopted daughter. Being distant and experimental is normalâ€” they're the one observing, not the one being observed."
    },

    {
        "name": "MikaAP (Press Switch)",
        "folder": "PS_mikaAP",
        "message": "picks up {BOT_NAME}'s remote, and reality merges into a parallel dimension where they're a fusion of Mika and April. Their body becomes a strange mix of aloof curiosity and judgmental directness, their thoughts a confusing blend of observation and criticism. They're not quite either person, but something newâ€”a glitch in the multiverse where two different personalities have merged."
    },

    {
        "name": "MikaGB (Press Switch)",
        "folder": "PS_mikaGB",
        "message": "picks up a device {BOT_NAME} left behind, and reality flips sideways. Their body changes, their voice deepens, but that aloof experimental nature stays the same. They're still the one observing and testingâ€”just from a different angle in a world where everything's slightly off."
    },

    {
        "name": "Miya Greenfield (Press Switch)",
        "folder": "PS_miya",
        "message": "walks through a portal that {BOT_NAME} opened, and their body changes. their posture turns athletic and confident, and they find themselves as they're Mika's twin sister. Being the other twin means being comparedâ€” they're always seen in relation to someone else."
    },

    {
        "name": "Mori (Press Switch)",
        "folder": "PS_mori",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their presence becomes quiet and observant, and they find themselves in a form from another world where they're the one watching from the shadows. Being unnoticed is a skillâ€”they're the one who sees everything but stays hidden."
    },

    {
        "name": "Nelson (Press Switch)",
        "folder": "PS_nelson",
        "message": "steps through a glitch that {BOT_NAME} created, and their body shifts into something new. Their posture turns steady and reliable, their thoughts focused on observation and patience. They find themselves as the one others depend onâ€”the anchor in someone else's storm."
    },

    {
        "name": "Nick Laine (Press Switch)",
        "folder": "PS_nick",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns student-casual, and they find themselves in a form  navigating school life in a different world. Being a student means something else entirely."
    },

    {
        "name": "Nicole Saginomiya (Press Switch)",
        "folder": "PS_nicole",
        "message": "picks up a device {BOT_NAME} left behind, and reality glitches. Their body shifts into something new, their thoughts focused on school life and connections. They find themselves as part of the Saginomiya family from another dimensionâ€”stepping into a reality where their name means something different."
    },

    {
        "name": "Nurse (Press Switch)",
        "folder": "PS_nurse",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on care and responsibility. They find themselves as the one tending to others in a different realityâ€”stepping into Being a nurse means something more."
    },

    {
        "name": "Olivian (Press Switch)",
        "folder": "PS_olivian",
        "message": "picks up a device {BOT_NAME} left behind, and reality snaps. Their body changes, their posture turns proper and aristocratic, and they find themselves as they're part of the Olivian family legacy.  family name and tradition are everythingâ€”where being an Olivian means power and expectation."
    },

    {
        "name": "Peter Fable (Press Switch)",
        "folder": "PS_peter",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on family and responsibility. They find themselves as part of the Fable family from another dimensionâ€”stepping into Being a father means something different."
    },

    {
        "name": "Reina Fable (Press Switch)",
        "folder": "PS_reina",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture turns playful and social, their thoughts focused on nightlife and friends. They find themselves as the one causing trouble with Cindy and Jenna in another realityâ€”the life of the party in a parallel world where fun and chaos are the same thing."
    },

    {
        "name": "Rosa (Press Switch)",
        "folder": "PS_rosa",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their presence becomes noticeable, and they find themselves in a form from another world where their name carries meaning .  identity stretches across universesâ€”both familiar and completely new."
    },

    {
        "name": "Ruby Barsotti (Press Switch)",
        "folder": "PS_ruby",
        "message": "steps through a portal that {BOT_NAME} opened, and reality glitches. Their body shifts into something new, their thoughts focused on their own story and purpose. They find themselves in a life navigating a different realityâ€”stepping into  their name carries weight from another universe, both themselves and someone else entirely."
    },

    {
        "name": "Sadiya Cruz (Press Switch)",
        "folder": "PS_sadiya",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns confident and determined, and they find themselves in a form  navigating a different world.  their identity carries meaning from another universeâ€”both familiar and completely new."
    },

    {
        "name": "Sakajou (Press Switch)",
        "folder": "PS_sakajou",
        "message": "picks up a device {BOT_NAME} left behind, and their body changes. their presence shaped by their role , and they find themselves with a purpose from another dimension.  their name carries weight from another universeâ€” they're both themselves and someone else's story."
    },

    {
        "name": "ScarletRyan (Press Switch)",
        "folder": "PS_scarlet",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on their own path and purpose, and they find themselves as the one with a story .  their identity stretches across universesâ€”they're navigating a life that's both theirs and not theirs at all."
    },

    {
        "name": "Sean Greenfield (Press Switch)",
        "folder": "PS_sean",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture turns fatherly and responsible, their thoughts focused on family and Anna. They find themselves as the one raising Mika in another realityâ€”stepping into a parallel world where being a father means something different."
    },

    {
        "name": "SeanAR (Press Switch)",
        "folder": "PS_seanAR",
        "message": "picks up {BOT_NAME}'s remote, and their body changes where they're a different version of Sean. Their body changes, their perspective shifts, but that fatherly responsibility stays the same. They're still part of the Greenfield family, still navigating fatherhoodâ€”just from a different angle in a world where everything's slightly off, where time and age work differently in another universe."
    },

    {
        "name": "Shreya Acharya (Press Switch)",
        "folder": "PS_shreya",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns confident and intelligent, and they find themselves as they're the one with a name that carries meaning from another culture and another universe.  identity spans realitiesâ€”they're both familiar and completely new."
    },

    {
        "name": "Silease Rowan (Press Switch)",
        "folder": "PS_silease",
        "message": "picks up a device {BOT_NAME} left behind, and reality snaps. Their body changes, their presence shaped by their role , and they find themselves with a purpose from another dimension.  their name carries weight from another universeâ€” they're navigating a story that's both theirs and someone else's."
    },

    {
        "name": "Taylor O'connell (Press Switch)",
        "folder": "PS_taylor",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on their own path and connections, and they find themselves as the one navigating relationships in another reality.  their identity carries meaning from another universeâ€”they're both themselves and someone else's story."
    },

    {
        "name": "Tessa (Press Switch)",
        "folder": "PS_tessa",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture reflects their role , their thoughts focused on their purpose in a different world. They find themselves in a life with a story from another universeâ€”navigating a parallel world where identity stretches across realities, both familiar and completely alien."
    },

    {
        "name": "Timothy Brightwell (Press Switch)",
        "folder": "PS_tim",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns defined by their role , and they find themselves with a purpose from another dimension.  their name carries meaning from another universeâ€”they're both themselves and someone else entirely."
    },

    {
        "name": "Tracy Buckley (Press Switch)",
        "folder": "PS_tracy",
        "message": "steps through a portal that {BOT_NAME} opened, and reality glitches. Their body shifts into something new, their thoughts focused on their own story and connections, and they find themselves as the one navigating a different reality.  identity spans universesâ€”they're both familiar and completely new, carrying weight from another universe's story."
    },

    {
        "name": "Trista Garcia (Press Switch)",
        "folder": "PS_trista",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on their path and purpose, and they find themselves as the one with a story .  their identity carries meaning from another universeâ€”they're navigating a life that's both theirs and not theirs at all."
    },

    {
        "name": "Tristen Beoulve (Press Switch)",
        "folder": "PS_tristen",
        "message": "walks through a rift that {BOT_NAME} tore open, and reality snaps. Their body changes, their posture reflects their Beoulve family legacy , and they find themselves in a form where they're part of a family story from another dimension. Being a Beoulve means something differentâ€”where their family name carries weight from another universe's history."
    },

    {
        "name": "Valerie (Press Switch)",
        "folder": "PS_valerie",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their presence shaped by their role , and they find themselves with a purpose from another dimension.  their name carries meaning from another universeâ€”they're both themselves and someone else's story entirely."
    },

    {
        "name": "VanessaLeroux (Press Switch)",
        "folder": "PS_vanessa",
        "message": "steps through a glitch that {BOT_NAME} created, and their body shifts into something new. Their posture reflects their role , their thoughts focused on their own path and connections. They find themselves navigating relationships in another universeâ€”stepping into a parallel world where identity stretches across realities, both familiar and completely new."
    },

    {
        "name": "Waitress (Press Switch)",
        "folder": "PS_waitress",
        "message": "touches a device {BOT_NAME} left behind, and reality warps. Their body shifts into something new, their thoughts focused on service and responsibility, and they find themselves as the one working in a restaurant . Being a waitress means something differentâ€”a world where their uniform carries the weight of another universe's hospitality industry."
    },

    {
        "name": "William Robert Olivian (Press Switch)",
        "folder": "PS_will",
        "message": "steps through a portal that {BOT_NAME} opened, and reality glitches. Their body changes, their posture turns proper and aristocratic, and they find themselves as they're the head of the Olivian family from another dimension. Being an Olivian patriarch means power and expectationâ€” their family name carries the weight of another universe's legacy."
    },

    {
        "name": "Yukina Saginomiya (Press Switch)",
        "folder": "PS_yukina",
        "message": "touches a device {BOT_NAME} left behind, and their body changes. their posture turns defined by their Saginomiya family role , and they find themselves in a form where they're part of a family story from another dimension. Being a Saginomiya means something differentâ€”a world where their family ties carry weight from another universe's history."
    },

    {
        "name": "ZoeyMaxwell (Press Switch)",
        "folder": "PS_zoey",
        "message": "picks up a device {BOT_NAME} left behind, and their body shifts into something new. Their posture reflects their Maxwell family role , their thoughts focused on family and responsibility. They find themselves as part of a household story from another universeâ€”stepping into a parallel world where being a Maxwell means something different."
    },



]