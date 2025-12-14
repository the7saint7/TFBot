import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##Uma Musume

    {
        "name": "Admire-Groove (Uma Musume)",
        "folder": "UMA_adgroove",
        "message": "touches a shimmering racing ribbon that {BOT_NAME} left behind, and a surge of competitive fire ignites in their chest. Their posture straightens with elegant determination, their movements become fluid and graceful, and they suddenly understand what it means to run with both beauty and powerâ€”every stride a work of art, every race a performance."
    },

    {
        "name": "Admire-Vega (Uma Musume)",
        "folder": "UMA_avega",
        "message": "picks up a star-patterned cloth that {BOT_NAME} left behind, and cosmic energy courses through them. Their movements become otherworldly smooth, their presence magnetic, and they suddenly understand what it means to shineâ€”not just run, but to be a star that others can't help but watch."
    },

    {
        "name": "Agnes-Digital (Uma Musume)",
        "folder": "UMA_adigital",
        "message": "picks up a digital watch that {BOT_NAME} left on the bench, and the screen flickers with strange symbols. A cool, analytical clarity washes over them, their thoughts organizing into precise patterns. They find themselves calculating angles, speeds, and strategiesâ€”seeing the track as a puzzle to solve, not just a path to run."
    },

    {
        "name": "Agnes-Tachyon (Uma Musume)",
        "folder": "UMA_atach",
        "message": "Insert messege here."
    },

    {
        "name": "Air-Groove (Uma Musume)",
        "folder": "UMA_agroove",
        "message": "steps into a pair of sleek racing shoes that {BOT_NAME} left behind, and the air itself seems to part for them. Their body lightens, their stride becomes impossibly smooth, and they feel the wind itself cheering them forwardâ€”running not just fast, but with an ethereal grace that makes speed look effortless."
    },

    {
        "name": "Air-Messiah (Uma Musume)",
        "folder": "UMA_amess",
        "message": "picks up a mechanical token that {BOT_NAME} left behind, and engineering precision flows through them. Their movements become calculated and efficient, their focus absolute, and they understand what it means to be machanâ€”running with the mechanical, unstoppable energy of something built for one purpose: to run perfectly, every time."
    },

    {
        "name": "Air-Shakur (Uma Musume)",
        "folder": "UMA_ash",
        "message": "Insert messege here."
    },

    {
        "name": "Almond-Eye (Uma Musume)",
        "folder": "UMA_almondeye",
        "message": "finds a pair of elegant glasses that {BOT_NAME} left behind, and when they put them on, the world sharpens into perfect focus. Their gaze becomes steady and observant, their movements precise and calculated. They see every detail, every opportunityâ€”the track becomes a canvas, and they're the artist who knows exactly where each stroke should go."
    },

    {
        "name": "Asahi-Rising (Uma Musume)",
        "folder": "UMA_asm",
        "message": "grabs a sunrise-colored ribbon that {BOT_NAME} left behind, and warmth spreads through them like dawn breaking. Their energy surges with new beginnings, their optimism becomes infectious, and they feel unstoppableâ€”like every race is a fresh start, every finish line a new horizon to chase."
    },

    {
        "name": "Asahikawa (Uma Musume)",
        "folder": "UMA_ash",
        "message": "touches a cold, crystalline object that {BOT_NAME} left behind, and a refreshing chill runs through them. Their mind clears, their focus sharpens, and they suddenly understand the beauty of running in crisp morning airâ€”every breath invigorating, every step clean and purposeful."
    },

    {
        "name": "Bamboo-Memory (Uma Musume)",
        "folder": "UMA_bamboo",
        "message": "touches a bamboo leaf that {BOT_NAME} left behind, and ancient strength flows into them. Their stance becomes unshakeable, their resolve deepens like roots, and they feel connected to something timelessâ€”running with the steady, unbreakable spirit of something that has weathered every storm."
    },

    {
        "name": "Believe (Uma Musume)",
        "folder": "UMA_believe",
        "message": "finds a small, glowing charm that {BOT_NAME} left behind, and unshakeable faith fills their heart. Their doubts vanish, their confidence becomes absolute, and they suddenly knowâ€”really knowâ€”that they can achieve anything. Every step forward feels like destiny, every challenge like a promise they're meant to keep."
    },

    {
        "name": "Biko-Pegasus (Uma Musume)",
        "folder": "UMA_bikop",
        "message": "Insert messege here."
    },

    {
        "name": "Biko-Pegasus (Uma Musume)",
        "folder": "UMA_bikop",
        "message": "finds a biwa string that {BOT_NAME} left behind, and musical precision flows through them. Their movements become rhythmic and calculated, their mind analytical, and they understand what it means to be Hayahideâ€”running with the methodical, intelligent energy of someone who studies every detail, turning racing into a precise art form."
    },

    {
        "name": "Biwa-Hayahide (Uma Musume)",
        "folder": "UMA_bikop",
        "message": "finds a biwa string that {BOT_NAME} left behind, and musical precision flows through them. Their movements become rhythmic and calculated, their mind analytical, and they understand what it means to be Hayahideâ€”running with the methodical, intelligent energy of someone who studies every detail, turning racing into a precise art form."
    },

    {
        "name": "Blast-Onepiece (Uma Musume)",
        "folder": "UMA_blast",
        "message": "grabs a pair of explosive-patterned gloves that {BOT_NAME} left behind, and raw power surges through their veins. Their muscles tense with explosive potential, their energy becomes volatile and intense, and they feel like they could shatter records with a single burstâ€”running not just fast, but with devastating force."
    },

    {
        "name": "Bubble-Gum-Fellow (Uma Musume)",
        "folder": "UMA_bubble",
        "message": "pops a piece of bubble gum that {BOT_NAME} left behind, and playful energy bubbles up inside them. Their movements become bouncy and light, their mood turns cheerful and carefree, and they suddenly want to make running funâ€”every race a game, every victory a celebration."
    },

    {
        "name": "Buena-Vista (Uma Musume)",
        "folder": "UMA_buena",
        "message": "steps into a pair of elegant boots that {BOT_NAME} left behind, and refined grace settles over them. Their posture becomes aristocratic, their movements sophisticated, and they understand what it means to run with classâ€”every stride elegant, every finish a statement of excellence."
    },

    {
        "name": "Byerley-Turk (Uma Musume)",
        "folder": "UMA_byerley",
        "message": "touches an ornate medallion that {BOT_NAME} left behind, and noble heritage awakens in their blood. Their bearing becomes regal, their determination fierce, and they feel the weight of legacyâ€”running not just for themselves, but for a lineage of champions."
    },

    {
        "name": "Calstone (Uma Musume)",
        "folder": "UMA_calstone",
        "message": "picks up a smooth, polished stone that {BOT_NAME} left behind, and unyielding strength solidifies in their core. Their resolve becomes rock-hard, their endurance limitless, and they understand what it means to be unbreakableâ€”running with the steady, relentless force of stone."
    },

    {
        "name": "Cesario (Uma Musume)",
        "folder": "UMA_cesario",
        "message": "finds a theatrical mask that {BOT_NAME} left behind, and dramatic flair takes over. Their movements become expressive and grand, their presence commanding, and they suddenly see racing as performanceâ€”every race a stage, every victory a standing ovation waiting to happen."
    },

    {
        "name": "Cheval-Grand (Uma Musume)",
        "folder": "UMA_cheval",
        "message": "grabs a golden bridle that {BOT_NAME} left behind, and noble power surges through them. Their stance becomes majestic, their will unbreakable, and they feel like royaltyâ€”running with the dignity and strength of a champion who knows their worth."
    },

    {
        "name": "Chrono-Genesis (Uma Musume)",
        "folder": "UMA_chrono",
        "message": "touches a timepiece that {BOT_NAME} left behind, and temporal awareness floods their mind. They see moments stretching ahead, understand the flow of time itself, and suddenly know exactly when to push, when to hold backâ€”running not just in the present, but with perfect timing across past, present, and future."
    },

    {
        "name": "Copano-Rickey (Uma Musume)",
        "folder": "UMA_copano",
        "message": "picks up a vibrant, colorful accessory that {BOT_NAME} left behind, and energetic charisma explodes through them. Their personality becomes magnetic, their enthusiasm infectious, and they suddenly want to make everyone smileâ€”running with joy, spreading happiness with every step."
    },

    {
        "name": "Curren-Bouquetd'or (Uma Musume)",
        "folder": "UMA_currenb",
        "message": "finds a sleek, dark ribbon that {BOT_NAME} left behind, and shadowy intensity wraps around them. Their focus becomes absolute, their determination dark and unyielding, and they understand what it means to run in the darknessâ€”finding strength where others see only absence."
    },

    {
        "name": "Curren-Chan (Uma Musume)",
        "folder": "UMA_currenc",
        "message": "touches a delicate, feminine charm that {BOT_NAME} left behind, and graceful elegance flows through them. Their movements become refined and beautiful, their presence gentle yet powerful, and they suddenly understand the art of running with both strength and grace."
    },

    {
        "name": "Daiichi-Ruby (Uma Musume)",
        "folder": "UMA_daiichi",
        "message": "grabs a precious gem that {BOT_NAME} left behind, and brilliant determination crystallizes in their heart. Their resolve becomes unbreakable, their spirit precious and rare, and they feel like something valuable being polished to perfectionâ€”every race making them shine brighter."
    },

    {
        "name": "Daitaku-Helios (Uma Musume)",
        "folder": "UMA_daitaku",
        "message": "picks up a sun-emblazoned token that {BOT_NAME} left behind, and solar energy courses through them. Their power becomes radiant, their presence bright and overwhelming, and they understand what it means to be a sunâ€”running with the unstoppable force of light itself."
    },

    {
        "name": "Daiwa-Scarlet (Uma Musume)",
        "folder": "UMA_daiwa",
        "message": "finds a crimson ribbon that {BOT_NAME} left behind, and passionate fire ignites in their soul. Their determination becomes fierce and unyielding, their will burns bright, and they suddenly know what it means to run with everythingâ€”every race a battle, every finish line a victory worth fighting for."
    },

    {
        "name": "Dantsu-Flame (Uma Musume)",
        "folder": "UMA_dantsu",
        "message": "touches a flickering flame that {BOT_NAME} left behind, and burning intensity consumes them. Their energy becomes volatile and hot, their passion explosive, and they feel like they could set the track on fireâ€”running with the destructive, beautiful force of flame."
    },

    {
        "name": "Daring-Heart (Uma Musume)",
        "folder": "UMA_daringt",
        "message": "Insert messege here."
    },

    {
        "name": "Daring-Tact (Uma Musume)",
        "folder": "UMA_daringh",
        "message": "grabs a bold, daring token that {BOT_NAME} left behind, and fearless energy surges through them. Their courage becomes absolute, their determination unbreakable, and they understand what it means to be daringâ€”running with the unstoppable, bold energy of someone who takes risks, makes bold moves, and never hesitates when victory is on the line."
    },

    {
        "name": "Darley-Arabian (Uma Musume)",
        "folder": "UMA_darley",
        "message": "grabs an ornate, exotic token that {BOT_NAME} left behind, and ancient desert strength awakens. Their endurance becomes legendary, their spirit untamed, and they understand what it means to run across endless horizonsâ€”every race a journey, every finish a new beginning."
    },

    {
        "name": "Dream-Journey (Uma Musume)",
        "folder": "UMA_dream",
        "message": "picks up a dreamcatcher that {BOT_NAME} left behind, and ethereal determination fills them. Their goals become clear as visions, their path illuminated, and they suddenly see the future they're running towardâ€”every step bringing them closer to the dream they're chasing."
    },

    {
        "name": "Duramente (Uma Musume)",
        "folder": "UMA_duramente",
        "message": "finds a heavy, solid weight that {BOT_NAME} left behind, and unshakeable strength anchors them. Their resolve becomes immovable, their will unbreakable, and they understand what it means to be durableâ€”running with the relentless, enduring force of something that never gives up."
    },

    {
        "name": "Durandal (Uma Musume)",
        "folder": "UMA_durandal",
        "message": "touches a legendary blade's hilt that {BOT_NAME} left behind, and heroic power surges through them. Their spirit becomes unbreakable, their determination legendary, and they feel like a warriorâ€”running not just to win, but to prove that some things are worth fighting for."
    },

    {
        "name": "Eishin-Flash (Uma Musume)",
        "folder": "UMA_eishin",
        "message": "grabs a lightning-bolt pin that {BOT_NAME} left behind, and electric speed courses through their veins. Their movements become impossibly fast, their reflexes lightning-quick, and they suddenly understand what it means to be a flashâ€”running so fast they leave light itself behind."
    },

    {
        "name": "El-Condor-Pasa (Uma Musume)",
        "folder": "UMA_el",
        "message": "picks up a feather that {BOT_NAME} left behind, and soaring freedom fills their heart. Their spirit becomes untamed, their movements graceful and powerful, and they understand what it means to flyâ€”running not on the ground, but soaring above it."
    },

    {
        "name": "Espoir-City (Uma Musume)",
        "folder": "UMA_espoir",
        "message": "finds a city-shaped charm that {BOT_NAME} left behind, and urban energy pulses through them. Their determination becomes metropolitan, their drive relentless, and they suddenly see the track as a city to conquerâ€”every race a street to master, every victory a district claimed."
    },

    {
        "name": "Fenomeno (Uma Musume)",
        "folder": "UMA_fenomeno",
        "message": "touches a mysterious, glowing object that {BOT_NAME} left behind, and phenomenal power awakens. Their abilities become extraordinary, their presence otherworldly, and they understand what it means to be a phenomenonâ€”running in ways that defy explanation."
    },

    {
        "name": "Fine-Motion (Uma Musume)",
        "folder": "UMA_fine",
        "message": "grabs a precision instrument that {BOT_NAME} left behind, and mechanical perfection flows through them. Their movements become flawlessly precise, their technique perfect, and they suddenly see running as an art formâ€”every motion calculated, every stride a masterpiece of efficiency."
    },

    {
        "name": "Fuji-Kiseki (Uma Musume)",
        "folder": "UMA_fuji",
        "message": "picks up a mountain-shaped token that {BOT_NAME} left behind, and unshakeable determination solidifies in their core. Their resolve becomes as immovable as a mountain, their spirit unbreakable, and they understand what it means to be a miracleâ€”running with the impossible strength of something that shouldn't exist but does."
    },

    {
        "name": "Furioso (Uma Musume)",
        "folder": "UMA_furioso",
        "message": "finds a wild, untamed ribbon that {BOT_NAME} left behind, and furious passion explodes through them. Their energy becomes wild and untamed, their determination fierce, and they suddenly understand what it means to run with furyâ€”every race a storm, every finish a tempest."
    },

    {
        "name": "Fusaichi-Pandora (Uma Musume)",
        "folder": "UMA_fusaichi",
        "message": "touches a winged medallion that {BOT_NAME} left behind, and legendary power awakens. Their spirit becomes mythical, their speed otherworldly, and they feel like a creature of legendâ€”running with the impossible grace and power of a pegasus taking flight."
    },

    {
        "name": "Gentildonna (Uma Musume)",
        "folder": "UMA_gentildonna",
        "message": "grabs an elegant, refined accessory that {BOT_NAME} left behind, and noble grace settles over them. Their movements become sophisticated and beautiful, their presence commanding yet gentle, and they understand what it means to be a ladyâ€”running with dignity, elegance, and unbreakable will."
    },

    {
        "name": "Godolphin-Barb (Uma Musume)",
        "folder": "UMA_godolphin",
        "message": "picks up a royal crest that {BOT_NAME} left behind, and aristocratic power flows through them. Their bearing becomes regal, their determination absolute, and they feel like nobilityâ€”running with the unshakeable confidence of someone born to lead."
    },

    {
        "name": "Gold-City (Uma Musume)",
        "folder": "UMA_goldc",
        "message": "finds a golden coin that {BOT_NAME} left behind, and urban ambition ignites in their heart. Their drive becomes relentless, their goals clear, and they suddenly see the track as a city of goldâ€”every race a chance to claim their fortune."
    },

    {
        "name": "Gold-Ship (Uma Musume)",
        "folder": "UMA_golds",
        "message": "touches a mischievous charm that {BOT_NAME} left behind, and playful chaos bubbles up inside them. Their personality becomes unpredictable, their antics legendary, and they suddenly understand what it means to be a shipâ€”sailing through races with wild, unpredictable energy that makes everyone watch."
    },

    {
        "name": "Gran-Alegria (Uma Musume)",
        "folder": "UMA_gran",
        "message": "touches an exotic, foreign token that {BOT_NAME} left behind, and international energy flows through them. Their movements become refined and worldly, their presence sophisticated, and they understand what it means to be from Algeriaâ€”running with the elegant, cultured energy of someone who has seen the world and brings that experience to every race."
    },

    {
        "name": "Grass-Wonder (Uma Musume)",
        "folder": "UMA_grass",
        "message": "touches a blade of grass that {BOT_NAME} left behind, and natural wonder flows through them. Their movements become gentle yet powerful, their spirit connected to the earth, and they understand what it means to be a wonderâ€”running with the simple, beautiful energy of nature itself, finding strength in the natural world and bringing that peace to every race."
    },

    {
        "name": "Haiseiko (Uma Musume)",
        "folder": "UMA_haiseiko",
        "message": "picks up a scholarly token that {BOT_NAME} left behind, and intellectual clarity sharpens their mind. Their strategy becomes calculated, their approach methodical, and they suddenly see racing as a problem to solveâ€”every race a puzzle, every victory a theorem proven."
    },

    {
        "name": "Haru-Urara (Uma Musume)",
        "folder": "UMA_haru",
        "message": "finds a spring flower that {BOT_NAME} left behind, and eternal optimism blooms in their heart. Their spirit becomes unbreakably cheerful, their determination endless, and they understand what it means to never give upâ€”running with the unstoppable joy of someone who finds happiness in every step."
    },

    {
        "name": "Hishi-Akebono (Uma Musume)",
        "folder": "UMA_hishia",
        "message": "finds a dawn-colored ribbon that {BOT_NAME} left behind, and morning energy awakens in their heart. Their spirit becomes fresh and new, their determination bright like sunrise, and they understand what it means to be Akebonoâ€”running with the beautiful, hopeful energy of someone who greets each day with renewed purpose and the promise of a new beginning."
    },

    {
        "name": "Hishi-Amazon (Uma Musume)",
        "folder": "UMA_hishiam",
        "message": "grabs a powerful, commanding token that {BOT_NAME} left behind, and amazonian strength awakens. Their presence becomes formidable, their will unbreakable, and they understand what it means to be an Amazonâ€”running with the fierce, unstoppable energy of a warrior, strong and assertive, never backing down from any challenge."
    },

    {
        "name": "Hishi-Miracle (Uma Musume)",
        "folder": "UMA_hishim",
        "message": "touches a rhythmic token that {BOT_NAME} left behind, and musical energy pulses through them. Their movements become a dance, their stride a rhythm, and they suddenly understand what it means to run with musicâ€”every race a performance, every step a beat."
    },

    {
        "name": "Hokko-Tarumae (Uma Musume)",
        "folder": "UMA_hokot",
        "message": "grabs a northern token that {BOT_NAME} left behind, and cold determination freezes in their core. Their resolve becomes unbreakable, their spirit unyielding, and they understand what it means to be from the northâ€”running with the fierce, unrelenting strength of winter itself."
    },

    {
        "name": "Ikuno-Dictus (Uma Musume)",
        "folder": "UMA_ikuno",
        "message": "picks up a decisive token that {BOT_NAME} left behind, and unwavering determination solidifies. Their choices become clear, their path certain, and they suddenly know exactly what they wantâ€”running with the absolute certainty of someone who has made their decision."
    },

    {
        "name": "Inari-One (Uma Musume)",
        "folder": "UMA_inari",
        "message": "finds a fox-shaped charm that {BOT_NAME} left behind, and clever energy flows through them. Their mind becomes sharp and cunning, their movements sly, and they understand what it means to be cleverâ€”running not just with speed, but with the cleverness to always find the right path."
    },

    {
        "name": "Ines-Fujin (Uma Musume)",
        "folder": "UMA_ines",
        "message": "touches a wind token that {BOT_NAME} left behind, and airy grace fills them. Their movements become light as wind, their presence ethereal, and they suddenly understand what it means to be a wind godâ€”running with the unstoppable, invisible force of air itself."
    },

    {
        "name": "Jungle-Pocket (Uma Musume)",
        "folder": "UMA_junglep",
        "message": "grabs a wild, untamed token that {BOT_NAME} left behind, and jungle energy explodes through them. Their movements become primal and powerful, their spirit untamed, and they understand what it means to be wildâ€”running with the fierce, natural power of the jungle."
    },

    {
        "name": "K.S.-Miracle (Uma Musume)",
        "folder": "UMA_ks",
        "message": "picks up a miraculous charm that {BOT_NAME} left behind, and impossible energy surges through them. Their abilities become extraordinary, their victories seem like miracles, and they understand what it means to be a miracleâ€”running with the unstoppable, unbelievable energy of someone who makes the impossible happen, turning every race into something that defies expectations."
    },

    {
        "name": "Katsuragi-Ace (Uma Musume)",
        "folder": "UMA_katsuragi",
        "message": "picks up an ace card that {BOT_NAME} left behind, and winning energy surges through them. Their confidence becomes absolute, their skill unmatched, and they suddenly know they're the bestâ€”running with the unshakeable certainty of someone who always has the winning hand."
    },

    {
        "name": "Kawakami-Princess (Uma Musume)",
        "folder": "UMA_kawakami",
        "message": "finds a royal tiara that {BOT_NAME} left behind, and princess-like grace settles over them. Their movements become elegant and refined, their presence regal, and they understand what it means to be royaltyâ€”running with the dignity and beauty of a princess."
    },

    {
        "name": "King-Halo (Uma Musume)",
        "folder": "UMA_kingh",
        "message": "touches a crown that {BOT_NAME} left behind, and kingly power awakens. Their bearing becomes regal, their authority absolute, and they feel like a kingâ€”running with the unshakeable confidence of someone born to rule."
    },

    {
        "name": "Kitasan-Black (Uma Musume)",
        "folder": "UMA_kitasan",
        "message": "picks up a black, determined token that {BOT_NAME} left behind, and unyielding strength flows through them. Their resolve becomes absolute, their determination legendary, and they understand what it means to be Kitasanâ€”running with the unstoppable, powerful energy of someone who works harder than anyone, turning every challenge into a stepping stone toward greatness."
    },

    {
        "name": "Love-Only-You (Uma Musume)",
        "folder": "UMA_loves",
        "message": "grabs a heart-shaped charm that {BOT_NAME} left behind, and loving energy fills their heart. Their personality becomes warm and affectionate, their presence comforting, and they suddenly understand what it means to be lovedâ€”running with the joy of someone who spreads love with every step."
    },

    {
        "name": "Lucky-Lilac (Uma Musume)",
        "folder": "UMA_lucky",
        "message": "picks up a lucky charm that {BOT_NAME} left behind, and fortunate energy surrounds them. Their luck becomes legendary, their fortune endless, and they suddenly understand what it means to be luckyâ€”running with the unstoppable good fortune of someone who always lands on their feet."
    },

    {
        "name": "Manhattan-Cafe (Uma Musume)",
        "folder": "UMA_manhattan",
        "message": "finds a coffee cup that {BOT_NAME} left behind, and urban sophistication flows through them. Their personality becomes refined and cultured, their movements smooth, and they understand what it means to be cosmopolitanâ€”running with the elegant energy of a city that never sleeps."
    },

    {
        "name": "Maruzensky (Uma Musume)",
        "folder": "UMA_maruzensky",
        "message": "touches a starry token that {BOT_NAME} left behind, and stellar power awakens. Their presence becomes otherworldly, their speed cosmic, and they understand what it means to be a starâ€”running with the brilliant, unstoppable force of starlight itself."
    },

    {
        "name": "Marvelous-Sunday (Uma Musume)",
        "folder": "UMA_marvelouss",
        "message": "grabs a weekend charm that {BOT_NAME} left behind, and marvelous energy fills them. Their mood becomes perfect, their day ideal, and they suddenly understand what it means to have a marvelous Sundayâ€”running with the joy and relaxation of the perfect day."
    },

    {
        "name": "Matikanefukukitaru (Uma Musume)",
        "folder": "UMA_matikanefukukitaru",
        "message": "picks up an ornate, complex token that {BOT_NAME} left behind, and intricate power flows through them. Their movements become complex and beautiful, their presence multifaceted, and they understand what it means to be complexâ€”running with the beautiful, intricate patterns of something that can't be simplified."
    },

    {
        "name": "Matikanetannhauser (Uma Musume)",
        "folder": "UMA_matikanetannhauser",
        "message": "finds a legendary token that {BOT_NAME} left behind, and epic power awakens. Their spirit becomes legendary, their determination mythic, and they understand what it means to be a legendâ€”running with the unstoppable force of a story that must be told."
    },

    {
        "name": "Mayano-Top-Gun (Uma Musume)",
        "folder": "UMA_mayano",
        "message": "touches a military pin that {BOT_NAME} left behind, and elite determination solidifies. Their focus becomes absolute, their skill unmatched, and they suddenly know they're the best of the bestâ€”running with the precision and power of a top gun."
    },

    {
        "name": "Meisho-Doto (Uma Musume)",
        "folder": "UMA_meisho",
        "message": "grabs a compass that {BOT_NAME} left behind, and directional clarity fills them. Their path becomes clear, their destination certain, and they understand what it means to have directionâ€”running with the unshakeable certainty of someone who knows exactly where they're going."
    },

    {
        "name": "Mejiro-Ardan (Uma Musume)",
        "folder": "UMA_mejiroa",
        "message": "picks up a noble token that {BOT_NAME} left behind, and aristocratic grace settles over them. Their bearing becomes refined, their movements elegant, and they understand what it means to be nobleâ€”running with the dignity and strength of aristocracy."
    },

    {
        "name": "Mejiro-Bright (Uma Musume)",
        "folder": "UMA_mejirob",
        "message": "finds a bright, shining token that {BOT_NAME} left behind, and brilliant energy fills them. Their presence becomes radiant, their spirit bright, and they understand what it means to shineâ€”running with the unstoppable brightness of someone who lights up every room."
    },

    {
        "name": "Mejiro-Dober (Uma Musume)",
        "folder": "UMA_mejirod",
        "message": "touches a fierce token that {BOT_NAME} left behind, and powerful determination awakens. Their strength becomes formidable, their will unbreakable, and they understand what it means to be powerfulâ€”running with the fierce, unstoppable force of a dober."
    },

    {
        "name": "Mejiro-McQueen (Uma Musume)",
        "folder": "UMA_mejirom",
        "message": "grabs a legendary token that {BOT_NAME} left behind, and queenly power surges through them. Their authority becomes absolute, their presence commanding, and they feel like a queenâ€”running with the unshakeable confidence of someone born to rule."
    },

    {
        "name": "Mejiro-Palmer (Uma Musume)",
        "folder": "UMA_mejirop",
        "message": "picks up an elegant token that {BOT_NAME} left behind, and refined grace flows through them. Their movements become sophisticated, their presence polished, and they understand what it means to be elegantâ€”running with the beautiful, refined energy of someone who knows style."
    },

    {
        "name": "Mejiro-Ramonu (Uma Musume)",
        "folder": "UMA_mejirora",
        "message": "finds a unique token that {BOT_NAME} left behind, and individual power awakens. Their spirit becomes unique, their path their own, and they understand what it means to be differentâ€”running with the strength of someone who carves their own way."
    },

    {
        "name": "Mejiro-Ryan (Uma Musume)",
        "folder": "UMA_mejirory",
        "message": "touches a royal token that {BOT_NAME} left behind, and kingly energy fills them. Their bearing becomes regal, their determination absolute, and they feel like royaltyâ€”running with the unshakeable confidence of a king."
    },

    {
        "name": "Mihono-Bourbon (Uma Musume)",
        "folder": "UMA_mihono",
        "message": "grabs a refined token that {BOT_NAME} left behind, and sophisticated energy flows through them. Their movements become elegant and smooth, their presence refined, and they understand what it means to be bourbonâ€”running with the smooth, refined energy of something aged to perfection."
    },

    {
        "name": "Mr-C.B. (Uma Musume)",
        "folder": "UMA_mr",
        "message": "picks up a mysterious token that {BOT_NAME} left behind, and enigmatic power awakens. Their presence becomes mysterious, their identity unclear, and they understand what it means to be a mysteryâ€”running with the intriguing energy of someone whose true nature is hidden."
    },

    {
        "name": "Nakayama-Festa (Uma Musume)",
        "folder": "UMA_nakayama",
        "message": "finds a festive token that {BOT_NAME} left behind, and celebratory energy explodes through them. Their mood becomes joyful, their presence festive, and they understand what it means to celebrateâ€”running with the unstoppable joy of someone who makes every day a festival."
    },

    {
        "name": "Narita-Brian (Uma Musume)",
        "folder": "UMA_naritab",
        "message": "touches a strong token that {BOT_NAME} left behind, and powerful determination solidifies. Their strength becomes legendary, their will unbreakable, and they understand what it means to be strongâ€”running with the unstoppable force of someone who never backs down."
    },

    {
        "name": "Narita-Taishin (Uma Musume)",
        "folder": "UMA_naritat",
        "message": "finds a determined token that {BOT_NAME} left behind, and unwavering resolve awakens. Their spirit becomes unbreakable, their will absolute, and they understand what it means to be Taishinâ€”running with the fierce, unstoppable energy of someone who refuses to give up, whose determination is as strong as steel, and who proves their worth with every step."
    },

    {
        "name": "Narita-Top-Road (Uma Musume)",
        "folder": "UMA_ntr",
        "message": "grabs a road marker that {BOT_NAME} left behind, and pathfinding energy flows through them. Their direction becomes clear, their route certain, and they understand what it means to be the top roadâ€”running with the unshakeable confidence of someone who knows they're on the right path, the best path, the one that leads straight to victory."
    },

    {
        "name": "Neo-Universe (Uma Musume)",
        "folder": "UMA_neo",
        "message": "grabs a cosmic token that {BOT_NAME} left behind, and universal power awakens. Their presence becomes otherworldly, their scope infinite, and they understand what it means to be a universeâ€”running with the boundless, infinite energy of space itself."
    },

    {
        "name": "Nice-Nature (Uma Musume)",
        "folder": "UMA_nice",
        "message": "picks up a natural token that {BOT_NAME} left behind, and gentle strength flows through them. Their personality becomes kind and natural, their movements organic, and they understand what it means to be niceâ€”running with the gentle, unstoppable force of nature itself."
    },

    {
        "name": "Nishino-Flower (Uma Musume)",
        "folder": "UMA_nishino",
        "message": "finds a flower that {BOT_NAME} left behind, and blooming energy fills them. Their spirit becomes beautiful and growing, their presence floral, and they understand what it means to bloomâ€”running with the beautiful, growing energy of a flower reaching for the sun."
    },

    {
        "name": "No-Reason (Uma Musume)",
        "folder": "UMA_noreason",
        "message": "touches a chaotic token that {BOT_NAME} left behind, and unpredictable energy explodes through them. Their actions become random, their logic nonexistent, and they suddenly understand what it means to have no reasonâ€”running with the wild, chaotic energy of pure unpredictability."
    },

    {
        "name": "North-Flight (Uma Musume)",
        "folder": "UMA_north",
        "message": "grabs a northern token that {BOT_NAME} left behind, and cold determination freezes in their core. Their resolve becomes unbreakable, their spirit unyielding, and they understand what it means to fly northâ€”running with the fierce, unrelenting strength of someone who knows cold."
    },

    {
        "name": "Oguri-Cap (Uma Musume)",
        "folder": "UMA_oguri",
        "message": "picks up a cap that {BOT_NAME} left behind, and serious determination awakens. Their focus becomes absolute, their appetite for victory insatiable, and they understand what it means to be Oguriâ€”running with the unshakeable determination of someone who never stops improving."
    },

    {
        "name": "Orfevre (Uma Musume)",
        "folder": "UMA_orfevre",
        "message": "finds a golden token that {BOT_NAME} left behind, and brilliant power flows through them. Their presence becomes radiant, their spirit golden, and they understand what it means to be preciousâ€”running with the beautiful, valuable energy of something rare and perfect."
    },

    {
        "name": "Red-Desire (Uma Musume)",
        "folder": "UMA_red",
        "message": "touches a red token that {BOT_NAME} left behind, and passionate fire ignites in their soul. Their desire becomes absolute, their determination fierce, and they understand what it means to wantâ€”running with the unstoppable force of pure desire."
    },

    {
        "name": "Rhein-Kraft (Uma Musume)",
        "folder": "UMA_rhein",
        "message": "grabs a powerful token that {BOT_NAME} left behind, and raw strength surges through them. Their power becomes overwhelming, their force unstoppable, and they understand what it means to be kraftâ€”running with the unstoppable, raw energy of pure power."
    },

    {
        "name": "Rice-Shower (Uma Musume)",
        "folder": "UMA_rice",
        "message": "picks up a rice grain that {BOT_NAME} left behind, and celebratory energy fills them. Their mood becomes festive, their presence joyful, and they understand what it means to showerâ€”running with the unstoppable joy of someone who brings celebration wherever they go."
    },

    {
        "name": "Royce-and-Royce (Uma Musume)",
        "folder": "UMA_royce",
        "message": "finds an elegant token that {BOT_NAME} left behind, and refined grace settles over them. Their movements become sophisticated, their presence polished, and they understand what it means to be elegantâ€”running with the beautiful, refined energy of someone who knows luxury."
    },

    {
        "name": "Saint-Lite (Uma Musume)",
        "folder": "UMA_saint",
        "message": "touches a holy token that {BOT_NAME} left behind, and divine grace flows through them. Their presence becomes blessed, their movements sacred, and they understand what it means to be a saintâ€”running with the beautiful, divine energy of something blessed."
    },

    {
        "name": "Sakura-Bakushin-O (Uma Musume)",
        "folder": "UMA_sakurab",
        "message": "grabs a cherry blossom that {BOT_NAME} left behind, and explosive energy fills them. Their power becomes sudden and overwhelming, their presence dramatic, and they understand what it means to be explosiveâ€”running with the sudden, powerful force of an explosion."
    },

    {
        "name": "Sakura-Chitose-O (Uma Musume)",
        "folder": "UMA_sakurac",
        "message": "touches a thousand-year cherry blossom that {BOT_NAME} left behind, and ancient beauty flows through them. Their presence becomes timeless and elegant, their movements graceful like falling petals, and they understand what it means to be Chitoseâ€”running with the beautiful, enduring energy of something that has lasted a thousand years, bringing that legacy to every race."
    },

    {
        "name": "Sakura-Chiyono-O (Uma Musume)",
        "folder": "UMA_sakurach",
        "message": "touches an elegant cherry blossom that {BOT_NAME} left behind, and refined beauty flows through them. Their movements become graceful and sophisticated, their presence elegant, and they understand what it means to be Chiyonoâ€”running with the beautiful, refined energy of someone who embodies grace and elegance, making every race a display of perfect form."
    },

    {
        "name": "Sakura-Laurel (Uma Musume)",
        "folder": "UMA_sakural",
        "message": "picks up a laurel branch that {BOT_NAME} left behind, and victorious energy flows through them. Their determination becomes absolute, their will to win unbreakable, and they understand what it means to wear laurelsâ€”running with the unstoppable energy of someone destined for victory."
    },

    {
        "name": "Samson-Big (Uma Musume)",
        "folder": "UMA_samson",
        "message": "finds a strength token that {BOT_NAME} left behind, and legendary power awakens. Their strength becomes unmatched, their will unbreakable, and they understand what it means to be strongâ€”running with the unstoppable force of someone whose strength is legendary."
    },

    {
        "name": "Satono-Crown (Uma Musume)",
        "folder": "UMA_satonoc",
        "message": "touches a crown that {BOT_NAME} left behind, and royal power surges through them. Their authority becomes absolute, their presence commanding, and they feel like royaltyâ€”running with the unshakeable confidence of someone born to wear a crown."
    },

    {
        "name": "Satono-Diamond (Uma Musume)",
        "folder": "UMA_satonod",
        "message": "grabs a diamond that {BOT_NAME} left behind, and brilliant determination crystallizes. Their resolve becomes unbreakable, their spirit precious, and they understand what it means to be a diamondâ€”running with the beautiful, unbreakable energy of something that has been forged under pressure."
    },

    {
        "name": "Seeking-the-Pearl (Uma Musume)",
        "folder": "UMA_seeking",
        "message": "picks up a pearl that {BOT_NAME} left behind, and searching energy fills them. Their determination becomes focused, their quest clear, and they understand what it means to seekâ€”running with the unstoppable drive of someone searching for something precious."
    },

    {
        "name": "Seiun-Sky (Uma Musume)",
        "folder": "UMA_seiun",
        "message": "finds a sky token that {BOT_NAME} left behind, and heavenly energy flows through them. Their spirit becomes boundless, their presence ethereal, and they understand what it means to be the skyâ€”running with the infinite, unstoppable energy of the heavens themselves."
    },

    {
        "name": "Shinko-Windy (Uma Musume)",
        "folder": "UMA_shinko",
        "message": "touches a wind token that {BOT_NAME} left behind, and airy power awakens. Their movements become light and fast, their presence breezy, and they understand what it means to be windyâ€”running with the unstoppable, invisible force of wind itself."
    },

    {
        "name": "Silence-Suzuka (Uma Musume)",
        "folder": "UMA_silence",
        "message": "grabs a silent token that {BOT_NAME} left behind, and calm focus settles over them. Their mind becomes perfectly still, their movements graceful and precise, and they understand what it means to be silenceâ€”running with the beautiful, perfect calm of someone who speaks through action alone."
    },

    {
        "name": "Sirius-Symboli (Uma Musume)",
        "folder": "UMA_sirius",
        "message": "picks up a star token that {BOT_NAME} left behind, and stellar power awakens. Their presence becomes brilliant, their speed cosmic, and they understand what it means to be Siriusâ€”running with the brightest, most unstoppable energy of the brightest star."
    },

    {
        "name": "Smart-Falcon (Uma Musume)",
        "folder": "UMA_smart",
        "message": "finds a falcon feather that {BOT_NAME} left behind, and sharp intelligence awakens. Their mind becomes razor-sharp, their movements precise, and they understand what it means to be smartâ€”running with the calculated, intelligent energy of someone who thinks faster than they move."
    },

    {
        "name": "Sounds-of-Earth (Uma Musume)",
        "folder": "UMA_sounds",
        "message": "touches an earth token that {BOT_NAME} left behind, and grounding energy flows through them. Their connection to the earth becomes absolute, their strength natural, and they understand what it means to be of the earthâ€”running with the steady, unbreakable force of the planet itself."
    },

    {
        "name": "Special-Week (Uma Musume)",
        "folder": "UMA_special",
        "message": "grabs a special token that {BOT_NAME} left behind, and unique determination awakens. Their spirit becomes special, their drive unmatched, and they understand what it means to be specialâ€”running with the unstoppable energy of someone who knows they're meant for greatness."
    },

    {
        "name": "Speed-Symboli (Uma Musume)",
        "folder": "UMA_speed",
        "message": "picks up a speed token that {BOT_NAME} left behind, and velocity itself courses through them. Their speed becomes absolute, their movements impossibly fast, and they understand what it means to be speedâ€”running with the unstoppable force of velocity itself."
    },

    {
        "name": "Stay-Gold (Uma Musume)",
        "folder": "UMA_stay",
        "message": "finds a golden token that {BOT_NAME} left behind, and eternal determination solidifies. Their resolve becomes unbreakable, their spirit timeless, and they understand what it means to stay goldâ€”running with the beautiful, enduring energy of something that never tarnishes."
    },

    {
        "name": "Still-in-Love (Uma Musume)",
        "folder": "UMA_still",
        "message": "touches a heart token that {BOT_NAME} left behind, and loving energy fills them. Their passion becomes endless, their devotion absolute, and they understand what it means to be in loveâ€”running with the unstoppable joy of someone who loves what they do."
    },

    {
        "name": "Super-Creek (Uma Musume)",
        "folder": "UMA_superc",
        "message": "grabs a water token that {BOT_NAME} left behind, and flowing power awakens. Their movements become fluid and unstoppable, their presence like a river, and they understand what it means to be a creekâ€”running with the steady, unbreakable force of water that always finds its way."
    },

    {
        "name": "Sweep-Tosho (Uma Musume)",
        "folder": "UMA_sweep",
        "message": "picks up a sweeping token that {BOT_NAME} left behind, and overwhelming energy fills them. Their power becomes absolute, their victory certain, and they understand what it means to sweepâ€”running with the unstoppable force of someone who wins everything."
    },

    {
        "name": "Symboli-Kris-S (Uma Musume)",
        "folder": "UMA_symbolik",
        "message": "finds a symbol token that {BOT_NAME} left behind, and symbolic power awakens. Their presence becomes meaningful, their actions significant, and they understand what it means to be a symbolâ€”running with the weight and power of something that represents more than itself."
    },

    {
        "name": "Symboli-Rudolf (Uma Musume)",
        "folder": "UMA_symbolir",
        "message": "touches a legendary token that {BOT_NAME} left behind, and mythic power surges through them. Their spirit becomes legendary, their determination absolute, and they understand what it means to be Rudolfâ€”running with the unstoppable force of a legend that will never die."
    },

    {
        "name": "T.M.-Opera-O (Uma Musume)",
        "folder": "UMA_tmo",
        "message": "grabs an opera token that {BOT_NAME} left behind, and dramatic power awakens. Their presence becomes theatrical, their movements grand, and they understand what it means to be an operaâ€”running with the dramatic, beautiful energy of a performance that moves hearts."
    },

    {
        "name": "Taiki-Shuttle (Uma Musume)",
        "folder": "UMA_taiki",
        "message": "grabs a shuttle token that {BOT_NAME} left behind, and rapid energy fills them. Their speed becomes incredible, their movements precise, and they understand what it means to be a shuttleâ€”running with the fast, efficient energy of something that moves between places instantly."
    },

    {
        "name": "Tamamo-Cross (Uma Musume)",
        "folder": "UMA_tamamo",
        "message": "picks up a cross token that {BOT_NAME} left behind, and determined power awakens. Their resolve becomes unbreakable, their will absolute, and they understand what it means to crossâ€”running with the unstoppable determination of someone who will cross any finish line."
    },

    {
        "name": "Tanino-Gimlet (Uma Musume)",
        "folder": "UMA_tanino",
        "message": "finds a sharp token that {BOT_NAME} left behind, and piercing energy flows through them. Their focus becomes razor-sharp, their movements precise, and they understand what it means to be a gimletâ€”running with the sharp, unstoppable energy of something that cuts through everything."
    },

    {
        "name": "Tap-Dance-City (Uma Musume)",
        "folder": "UMA_tap",
        "message": "touches a tap token that {BOT_NAME} left behind, and rhythmic energy fills them. Their movements become a dance, their stride musical, and they understand what it means to tap danceâ€”running with the joyful, rhythmic energy of someone who makes every step a performance."
    },

    {
        "name": "Tokai-Teio (Uma Musume)",
        "folder": "UMA_tokai",
        "message": "picks up an emperor token that {BOT_NAME} left behind, and imperial power surges through them. Their authority becomes absolute, their presence commanding, and they understand what it means to be an emperorâ€”running with the unshakeable confidence of someone born to rule everything."
    },

    {
        "name": "Tosen-Jordan (Uma Musume)",
        "folder": "UMA_tosen",
        "message": "finds a river token that {BOT_NAME} left behind, and flowing energy courses through them. Their movements become fluid and unstoppable, their presence like a river, and they understand what it means to be a riverâ€”running with the steady, unbreakable force of water that never stops."
    },

    {
        "name": "Transcend (Uma Musume)",
        "folder": "UMA_transcend",
        "message": "touches a transcendent token that {BOT_NAME} left behind, and otherworldly power awakens. Their abilities become beyond normal, their presence ethereal, and they understand what it means to transcendâ€”running with the unstoppable energy of something that has moved beyond limits."
    },

    {
        "name": "Tsurumaru-Tsuyoshi (Uma Musume)",
        "folder": "UMA_tsurumaru",
        "message": "grabs a crane token that {BOT_NAME} left behind, and graceful power flows through them. Their movements become elegant and beautiful, their presence refined, and they understand what it means to be a craneâ€”running with the beautiful, graceful energy of something that moves like poetry."
    },

    {
        "name": "Twin-Turbo (Uma Musume)",
        "folder": "UMA_twin",
        "message": "picks up a turbo token that {BOT_NAME} left behind, and explosive speed awakens. Their velocity becomes incredible, their power overwhelming, and they understand what it means to be turboâ€”running with the unstoppable, explosive force of something that moves at impossible speeds."
    },

    {
        "name": "Uma-Musume (Uma Musume)",
        "folder": "UMA_uma",
        "message": "finds a generic token that {BOT_NAME} left behind, and racing spirit awakens. Their determination becomes clear, their purpose certain, and they understand what it means to be an Uma Musumeâ€”running with the unstoppable energy of someone born to race."
    },

    {
        "name": "Venus-Paques (Uma Musume)",
        "folder": "UMA_venus",
        "message": "touches a venus token that {BOT_NAME} left behind, and beautiful power flows through them. Their presence becomes radiant, their movements graceful, and they understand what it means to be Venusâ€”running with the beautiful, unstoppable energy of the goddess of beauty herself."
    },

    {
        "name": "Verxina (Uma Musume)",
        "folder": "UMA_verxina",
        "message": "grabs a unique token that {BOT_NAME} left behind, and individual power awakens. Their spirit becomes their own, their path unique, and they understand what it means to be Verxinaâ€”running with the strength of someone who is completely themselves."
    },

    {
        "name": "Vivlos (Uma Musume)",
        "folder": "UMA_vivlos",
        "message": "picks up a vibrant token that {BOT_NAME} left behind, and living energy explodes through them. Their presence becomes vibrant and alive, their spirit unbreakable, and they understand what it means to liveâ€”running with the unstoppable energy of someone who is fully alive."
    },

    {
        "name": "Vodka (Uma Musume)",
        "folder": "UMA_vodka",
        "message": "finds a strong token that {BOT_NAME} left behind, and powerful energy flows through them. Their strength becomes formidable, their will unbreakable, and they understand what it means to be strongâ€”running with the unstoppable force of something that has been distilled to perfection."
    },

    {
        "name": "Win-Variation (Uma Musume)",
        "folder": "UMA_win",
        "message": "touches a winning token that {BOT_NAME} left behind, and victorious energy fills them. Their determination becomes absolute, their will to win unbreakable, and they understand what it means to winâ€”running with the unstoppable energy of someone who knows they will win."
    },

    {
        "name": "Winning-Ticket (Uma Musume)",
        "folder": "UMA_winning",
        "message": "grabs a ticket token that {BOT_NAME} left behind, and lucky determination awakens. Their fortune becomes certain, their victory guaranteed, and they understand what it means to have a winning ticketâ€”running with the unstoppable confidence of someone who knows they hold the winning hand."
    },

    {
        "name": "Wonder-Acute (Uma Musume)",
        "folder": "UMA_wonder",
        "message": "picks up an acute token that {BOT_NAME} left behind, and sharp energy flows through them. Their focus becomes razor-sharp, their movements precise, and they understand what it means to be acuteâ€”running with the sharp, unstoppable energy of someone who sees every angle."
    },

    {
        "name": "Yaeno-Muteki (Uma Musume)",
        "folder": "UMA_yaeno",
        "message": "finds an invincible token that {BOT_NAME} left behind, and unbeatable power awakens. Their strength becomes absolute, their victory certain, and they understand what it means to be mutekiâ€”running with the unstoppable force of someone who is truly invincible."
    },

    {
        "name": "Yamanin-Zephyr (Uma Musume)",
        "folder": "UMA_yamanin",
        "message": "touches a wind token that {BOT_NAME} left behind, and airy power flows through them. Their movements become light and fast, their presence breezy, and they understand what it means to be a zephyrâ€”running with the gentle, unstoppable force of a perfect breeze."
    },

    {
        "name": "Yukino-Bijin (Uma Musume)",
        "folder": "UMA_yukino",
        "message": "grabs a beauty token that {BOT_NAME} left behind, and beautiful energy fills them. Their presence becomes radiant, their movements graceful, and they understand what it means to be a bijinâ€”running with the beautiful, unstoppable energy of someone who is truly beautiful."
    },

    {
        "name": "Zenno-Rob-Roy (Uma Musume)",
        "folder": "UMA_zenno",
        "message": "picks up a royal token that {BOT_NAME} left behind, and noble power awakens. Their bearing becomes regal, their determination absolute, and they understand what it means to be royalâ€”running with the unshakeable confidence of someone born to nobility."
    },

]