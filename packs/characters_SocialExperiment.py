import os

# Bot name from environment, defaults to "Syn" for backwards compatibility
BOT_NAME = os.getenv("TFBOT_NAME", "Syn").strip() or "Syn"

# Helper function to replace bot name placeholder in messages
def _format_message(message: str) -> str:
    """Replace {BOT_NAME} placeholder with actual bot name."""
    return message.replace("{BOT_NAME}", BOT_NAME)

TF_CHARACTERS = [

    ##ST - Scenarios - Social Experiment

    {
        "name": "AbbyCornelia (Social Experiment)",
        "folder": "SE_abby_cornelia",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, Abby's head detaching from its original body in a flash of light. When the world stops spinning, Abby's head has landed on Cornelia's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "Akari (Social Experiment)",
        "folder": "SE_akari",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "AlexBouncer (Social Experiment)",
        "folder": "SE_alex_bouncer",
        "message": "picks up a strange remote that {BOT_NAME} left behind, and when they press a button, Alex's head detaches and floats over to Bouncer's body. The connection is seamless but wrongâ€”Alex's head sitting on Bouncer's shoulders, creating a person that shouldn't exist."
    },

    {
        "name": "AmyYuuna (Social Experiment)",
        "folder": "SE_amy_yuuna",
        "message": "activates {BOT_NAME}'s device by accident, and Amy's head pops off with a strange 'pop' sound. It floats through the air before landing on Yuuna's waiting shoulders. Now they're Amy's head on Yuuna's body, trying to make sense of this bizarre new reality."
    },

    {
        "name": "Avae (Social Experiment)",
        "folder": "SE_avae",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "BradKiyoshi (Social Experiment)",
        "folder": "SE_brad_kiyoshi",
        "message": "picks up a strange remote that {BOT_NAME} left behind, and when they press a button, Brad's head detaches and floats over to Kiyoshi's body. The connection is seamless but wrongâ€”Brad's head sitting on Kiyoshi's shoulders, creating a person that shouldn't exist."
    },

    {
        "name": "Carrie (Social Experiment)",
        "folder": "SE_carrieuniform",
        "message": "materializes after {BOT_NAME}'s device activates, their form solidifying from the experimental energy. They have no past to remember, no history to claimâ€”just the present moment and the knowledge that they're something new."
    },

    {
        "name": "CarrieJack (Social Experiment)",
        "folder": "SE_carrie_jack",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Carrie's head is sitting on Jack's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Carrie's face, Jack's form."
    },

    {
        "name": "ClausMichelle (Social Experiment)",
        "folder": "SE_claus_michelle",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Claus's head is sitting on Michelle's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Claus's face, Michelle's form."
    },

    {
        "name": "ConnieKyoko (Social Experiment)",
        "folder": "SE_connie_kyoko",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, Connie's head detaching from its original body in a flash of light. When the world stops spinning, Connie's head has landed on Kyoko's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "CorneliaAbby (Social Experiment)",
        "folder": "SE_cornelia_abby",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Cornelia's head is sitting on Abby's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Cornelia's face, Abby's form."
    },

    {
        "name": "EmilyLeona (Social Experiment)",
        "folder": "SE_emily_leona",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Emily's head floats free for a moment before settling onto Leona's shoulders, while Emily's thoughts flood their mind. They're Emily's head now, sitting on top of Leona's body like an ill-fitting combination."
    },

    {
        "name": "EricMaria (Social Experiment)",
        "folder": "SE_eric_maria",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Eric's head is sitting on Maria's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Eric's face, Maria's form."
    },

    {
        "name": "FlaviaMel (Social Experiment)",
        "folder": "SE_flavia_mel",
        "message": "activates {BOT_NAME}'s device by accident, and Flavia's head pops off with a strange 'pop' sound. It floats through the air before landing on Mel's waiting shoulders. Now they're Flavia's head on Mel's body, trying to make sense of this bizarre new reality."
    },

    {
        "name": "HollySandra (Social Experiment)",
        "folder": "SE_holly_sandra",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Holly's head floats free for a moment before settling onto Sandra's shoulders, while Holly's thoughts flood their mind. They're Holly's head now, sitting on top of Sandra's body like an ill-fitting combination."
    },

    {
        "name": "IreneLaura (Social Experiment)",
        "folder": "SE_irene_laura",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Irene's head is sitting on Laura's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Irene's face, Laura's form."
    },

    {
        "name": "IsabelleMaurice (Social Experiment)",
        "folder": "SE_isabelle_maurice",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, Isabelle's head detaching from its original body in a flash of light. When the world stops spinning, Isabelle's head has landed on Maurice's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "JackCarrie (Social Experiment)",
        "folder": "SE_jack_carrie",
        "message": "witnesses {BOT_NAME}'s device split Jack and Carrie apart, heads and bodies floating separately. Before they can react, Jack's head lands on Carrie's body, and suddenly they're seeing the world through this mismatched combinationâ€”Jack's head, Carrie's body."
    },

    {
        "name": "John-GB1 (Social Experiment)",
        "folder": "SE_johnGB1",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "John-GB2 (Social Experiment)",
        "folder": "SE_johnGB2",
        "message": "emerges from {BOT_NAME}'s latest creationâ€”a person with no clear origin, their history a blank slate. They exist now, in this moment, created by chance and circumstance, ready to discover who they'll become."
    },

    {
        "name": "JohnKatrina (Social Experiment)",
        "folder": "SE_john_katrina",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, John's head detaching from its original body in a flash of light. When the world stops spinning, John's head has landed on Katrina's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "KatrinaCarrie (Social Experiment)",
        "folder": "SE_katrinastalker",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "KatrinaJohn (Social Experiment)",
        "folder": "SE_katrina_john",
        "message": "activates {BOT_NAME}'s device by accident, and Katrina's head pops off with a strange 'pop' sound. It floats through the air before landing on John's waiting shoulders. Now they're Katrina's head on John's body, trying to make sense of this bizarre new reality."
    },

    {
        "name": "Katsumi (Social Experiment)",
        "folder": "SE_katsumi",
        "message": "emerges from {BOT_NAME}'s latest creationâ€”a person with no clear origin, their history a blank slate. They exist now, in this moment, created by chance and circumstance, ready to discover who they'll become."
    },

    {
        "name": "Kazuto (Social Experiment)",
        "folder": "SE_kazuto",
        "message": "materializes after {BOT_NAME}'s device activates, their form solidifying from the experimental energy. They have no past to remember, no history to claimâ€”just the present moment and the knowledge that they're something new."
    },

    {
        "name": "Kirika (Social Experiment)",
        "folder": "SE_kirika",
        "message": "emerges from {BOT_NAME}'s latest creationâ€”a person with no clear origin, their history a blank slate. They exist now, in this moment, created by chance and circumstance, ready to discover who they'll become."
    },

    {
        "name": "Kisaragi (Social Experiment)",
        "folder": "SE_kisaragi",
        "message": "emerges from {BOT_NAME}'s latest creationâ€”a person with no clear origin, their history a blank slate. They exist now, in this moment, created by chance and circumstance, ready to discover who they'll become."
    },

    {
        "name": "KiyoshiBrad (Social Experiment)",
        "folder": "SE_kiyoshi_brad",
        "message": "wakes up with a strange disorientation, their reflection showing Kiyoshi's face on Brad's body. {BOT_NAME}'s device must have separated Kiyoshi's head and Brad's body, then recombined themâ€”Kiyoshi's head sitting on Brad's shoulders, creating a confusing mix of identities that feels both familiar and alien."
    },

    {
        "name": "KyokoConnie (Social Experiment)",
        "folder": "SE_kyoko_connie",
        "message": "witnesses {BOT_NAME}'s device split Kyoko and Connie apart, heads and bodies floating separately. Before they can react, Kyoko's head lands on Connie's body, and suddenly they're seeing the world through this mismatched combinationâ€”Kyoko's head, Connie's body."
    },

    {
        "name": "LauraIrene (Social Experiment)",
        "folder": "SE_laura_irene",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Laura's head floats free for a moment before settling onto Irene's shoulders, while Laura's thoughts flood their mind. They're Laura's head now, sitting on top of Irene's body like an ill-fitting combination."
    },

    {
        "name": "LeonaEmily (Social Experiment)",
        "folder": "SE_leona_emily",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, Leona's head detaching from its original body in a flash of light. When the world stops spinning, Leona's head has landed on Emily's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "MariaEric (Social Experiment)",
        "folder": "SE_maria_eric",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Maria's head floats free for a moment before settling onto Eric's shoulders, while Maria's thoughts flood their mind. They're Maria's head now, sitting on top of Eric's body like an ill-fitting combination."
    },

    {
        "name": "Marissa (Social Experiment)",
        "folder": "SE_marissa",
        "message": "emerges from {BOT_NAME}'s latest creationâ€”a person with no clear origin, their history a blank slate. They exist now, in this moment, created by chance and circumstance, ready to discover who they'll become."
    },

    {
        "name": "MauriceIsabelle (Social Experiment)",
        "folder": "SE_maurice_isabelle",
        "message": "witnesses {BOT_NAME}'s device split Maurice and Isabelle apart, heads and bodies floating separately. Before they can react, Maurice's head lands on Isabelle's body, and suddenly they're seeing the world through this mismatched combinationâ€”Maurice's head, Isabelle's body."
    },

    {
        "name": "MelFlavia (Social Experiment)",
        "folder": "SE_mel_flavia",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Mel's head floats free for a moment before settling onto Flavia's shoulders, while Mel's thoughts flood their mind. They're Mel's head now, sitting on top of Flavia's body like an ill-fitting combination."
    },

    {
        "name": "MichelleClaus (Social Experiment)",
        "folder": "SE_michelle_claus",
        "message": "witnesses {BOT_NAME}'s device split Michelle and Claus apart, heads and bodies floating separately. Before they can react, Michelle's head lands on Claus's body, and suddenly they're seeing the world through this mismatched combinationâ€”Michelle's head, Claus's body."
    },

    {
        "name": "MinaSayaka (Social Experiment)",
        "folder": "SE_mina_sayaka",
        "message": "wakes up with a strange disorientation, their reflection showing Mina's face on Sayaka's body. {BOT_NAME}'s device must have separated Mina's head and Sayaka's body, then recombined themâ€”Mina's head sitting on Sayaka's shoulders, creating a confusing mix of identities that feels both familiar and alien."
    },

    {
        "name": "Nitel (Social Experiment)",
        "folder": "SE_nitel",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "Noire (Social Experiment)",
        "folder": "SE_noire",
        "message": "finds themselves in a new form after {BOT_NAME}'s experiment, their past unclear but their present undeniable. They're someone new, created from the chaos of {BOT_NAME}'s devices, with fragments of memories that don't quite fit together."
    },

    {
        "name": "Paige (Social Experiment)",
        "folder": "SE_paige",
        "message": "materializes after {BOT_NAME}'s device activates, their form solidifying from the experimental energy. They have no past to remember, no history to claimâ€”just the present moment and the knowledge that they're something new."
    },

    {
        "name": "PhilaRita (Social Experiment)",
        "folder": "SE_phila_rita",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Phila's head floats free for a moment before settling onto Rita's shoulders, while Phila's thoughts flood their mind. They're Phila's head now, sitting on top of Rita's body like an ill-fitting combination."
    },

    {
        "name": "RachelVanessa (Social Experiment)",
        "folder": "SE_rachel_vanessa",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Rachel's head floats free for a moment before settling onto Vanessa's shoulders, while Rachel's thoughts flood their mind. They're Rachel's head now, sitting on top of Vanessa's body like an ill-fitting combination."
    },

    {
        "name": "Risako (Social Experiment)",
        "folder": "SE_risako",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "RitaPhila (Social Experiment)",
        "folder": "SE_rita_phila",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Rita's head floats free for a moment before settling onto Phila's shoulders, while Rita's thoughts flood their mind. They're Rita's head now, sitting on top of Phila's body like an ill-fitting combination."
    },

    {
        "name": "Saeko (Social Experiment)",
        "folder": "SE_saeko",
        "message": "materializes after {BOT_NAME}'s device activates, their form solidifying from the experimental energy. They have no past to remember, no history to claimâ€”just the present moment and the knowledge that they're something new."
    },

    {
        "name": "SandraHolly (Social Experiment)",
        "folder": "SE_sandra_holly",
        "message": "touches a device that {BOT_NAME} left behind, and reality splits. Sandra's head floats free for a moment before settling onto Holly's shoulders, while Sandra's thoughts flood their mind. They're Sandra's head now, sitting on top of Holly's body like an ill-fitting combination."
    },

    {
        "name": "SayakaAbby (Social Experiment)",
        "folder": "SE_sayaka_abby",
        "message": "witnesses {BOT_NAME}'s device split Sayaka and Abby apart, heads and bodies floating separately. Before they can react, Sayaka's head lands on Abby's body, and suddenly they're seeing the world through this mismatched combinationâ€”Sayaka's head, Abby's body."
    },

    {
        "name": "SayakaMina (Social Experiment)",
        "folder": "SE_sayaka_mina",
        "message": "feels a sudden shift as {BOT_NAME}'s experiment activates, Sayaka's head detaching from its original body in a flash of light. When the world stops spinning, Sayaka's head has landed on Mina's bodyâ€”a surreal combination that leaves them questioning which parts of themselves are truly theirs."
    },

    {
        "name": "Stephani (Social Experiment)",
        "folder": "SE_stephani",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "ToriZoey (Social Experiment)",
        "folder": "SE_tori_zoey",
        "message": "activates {BOT_NAME}'s device by accident, and Tori's head pops off with a strange 'pop' sound. It floats through the air before landing on Zoey's waiting shoulders. Now they're Tori's head on Zoey's body, trying to make sense of this bizarre new reality."
    },

    {
        "name": "Valerie (Social Experiment)",
        "folder": "SE_valerie",
        "message": "finds themselves in a new form after {BOT_NAME}'s experiment, their past unclear but their present undeniable. They're someone new, created from the chaos of {BOT_NAME}'s devices, with fragments of memories that don't quite fit together."
    },

    {
        "name": "VanessaRachel (Social Experiment)",
        "folder": "SE_vanessa_rachel",
        "message": "activates {BOT_NAME}'s device by accident, and Vanessa's head pops off with a strange 'pop' sound. It floats through the air before landing on Rachel's waiting shoulders. Now they're Vanessa's head on Rachel's body, trying to make sense of this bizarre new reality."
    },

    {
        "name": "Xell (Social Experiment)",
        "folder": "SE_xell",
        "message": "wakes up with no memory of who they were before, only knowing that {BOT_NAME}'s device changed them. They're a new person now, born from the experiment, trying to piece together an identity from the fragments left behind."
    },

    {
        "name": "YuunaAmy (Social Experiment)",
        "folder": "SE_yuuna_amy",
        "message": "picks up a strange remote that {BOT_NAME} left behind, and when they press a button, Yuuna's head detaches and floats over to Amy's body. The connection is seamless but wrongâ€”Yuuna's head sitting on Amy's shoulders, creating a person that shouldn't exist."
    },

    {
        "name": "ZoeyTori (Social Experiment)",
        "folder": "SE_zoey_tori",
        "message": "stumbles into {BOT_NAME}'s latest experimentâ€”a head-swapping device that separates heads from bodies. When the light fades, Zoey's head is sitting on Tori's body. Every movement feels wrong, every reflection a reminder of the mix-upâ€”Zoey's face, Tori's form."
    },



]