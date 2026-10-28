"""
Sentiment analysis of customer feedback comments (Negative / Okay / Positive).

Approach: a transparent, rule-based classifier built around a domain-specific lexicon
(car dealership: Sales / Service / Parts feedback, Australian English).

Steps
  1. Clean the text (remove emails, phone numbers, rego plates, "call me on ..." requests)
  2. Match multi-word phrases first, then single words, from the lexicon
  3. Handle negation ("no dramas", "didn't call"), intensifiers and ALL-CAPS
  4. Handle sarcasm ("Brilliant. Another cancelled appointment. Cheers for that.")
  5. Combine positive and negative evidence:
       strong positive only -> Positive | strong negative only -> Negative
       both present in similar amounts, or neither/weak -> Okay

Only the 'Comment' text is used to classify. The Rating/Score column is NOT used.
Blank comments stay blank in the output (there is no text to classify).
"""
import re
import sys
import pandas as pd

IN_CSV = "combined_datasets/merged_all_feedback_cleaned.csv"
OUT_CSV = "combined_datasets/merged_all_feedback_cleaned_withSentAnaly.csv"
NEW_COL = "Sentiment from Comment"

# --------------------------------------------------------------------------
# 1. LEXICON  (weights: +/-2 strong, +/-1 moderate, +/-0.5 mild)
# --------------------------------------------------------------------------
POS_WORDS = {
    # strong
    "excellent": 2, "fantastic": 2, "outstanding": 2, "amazing": 2, "superb": 2, "exceptional": 2,
    "brilliant": 2, "wonderful": 2, "awesome": 2, "legend": 2, "ripper": 2, "flawless": 2,
    "delight": 2, "delightful": 2, "impressed": 1.5, "impressive": 1.5, "stoked": 2, "chuffed": 2,
    "thrilled": 2, "perfect": 2, "seamless": 2, "love": 1.5, "loved": 1.5, "recommend": 1.5,
    "recommended": 1.5, "recommending": 1.5, "fabulous": 2, "terrific": 2, "magic": 1.5, "gold": 1.5,
    "best": 1.5, "pleasure": 1.5,
    # moderate
    "great": 1.5, "good": 1, "nice": 1, "lovely": 1.5, "friendly": 1.5, "helpful": 1.5, "kind": 1.2,
    "kindness": 1.2, "warm": 1, "warmly": 1, "welcoming": 1.5, "welcome": 1, "welcomed": 1,
    "polite": 1.2, "courteous": 1.5, "cheerful": 1.5, "pleasant": 1.2, "pleasantly": 1.2,
    "patient": 1.2, "professional": 1.2, "efficient": 1.5, "prompt": 1.5, "promptly": 1.5,
    "quick": 1, "quickly": 1, "fast": 1, "speedy": 1.2, "swift": 1.2, "smooth": 1.5, "smoothly": 1.5,
    "easy": 1.2, "easier": 1.2, "straightforward": 1.2, "simple": 1, "painless": 1.5, "hassle-free": 1.5,
    "effortless": 1.5, "clear": 1, "clearly": 0.5, "honest": 1.5, "genuine": 1.5, "genuinely": 1,
    "transparent": 1.2, "fair": 1, "reasonable": 1, "affordable": 1, "cheaper": 1, "bargain": 1.5,
    "value": 0.8, "lower": 0.8, "understood": 1.2, "explained": 1, "explaining": 1, "explain": 0.5,
    "informed": 1, "organised": 1, "organized": 1, "orderly": 1, "tidy": 1, "thorough": 1.2,
    "thoroughly": 1, "pleased": 1.5, "happy": 1.5, "happier": 1.5, "satisfied": 1.2, "comfortable": 1,
    "relaxed": 1, "relieved": 1, "reassuring": 1.5, "reassured": 1.5, "appreciate": 1.5,
    "appreciated": 1.5, "grateful": 1.5, "thank": 1, "thanks": 1, "thankyou": 1, "thx": 1, "ta": 0.5,
    "cheers": 0.5, "approachable": 1.5, "obliging": 1.5, "accommodating": 1.5, "attentive": 1.5,
    "knowledgeable": 1.5, "accurate": 1, "reliable": 1.5, "trustworthy": 1.5, "caring": 1.5,
    "considerate": 1.5, "respectful": 1.5, "genuine": 1.5, "helped": 1, "assisted": 1, "smile": 1,
    "smiles": 1, "smiling": 1, "smiled": 1, "greeted": 0.8, "treated": 0.3, "sorted": 1, "fixed": 0.8,
    "resolved": 1.2, "handy": 1, "sweet": 1, "stress-free": 1.5, "worry-free": 1.5, "wow": 1,
    "refreshing": 1.2, "rare": 0.8, "gem": 1.5, "ease": 1, "recommendable": 1.5, "valued": 1.2,
    "comfy": 1, "calm": 0.8, "calmly": 0.8, "responsive": 1.5, "proactive": 1.5, "diligent": 1.5,
    "tops": 1.2, "top": 1, "ideal": 1.2, "beautiful": 1.5, "beautifully": 1.5, "cool": 0.8,
    "enjoyable": 1.5, "enjoyed": 1.5, "fun": 1, "spot-on": 1.5, "faultless": 2, "immaculate": 1.5,
    "spotless": 1.5, "sparkling": 1.2, "super": 1.2, "legendary": 2, "worth": 0.8, "worthwhile": 1,
    "refreshed": 1, "gladly": 1, "glad": 1, "fabulously": 2, "kudos": 1.5, "congrats": 1, "fairly": 0.3,
    "heaps": 0.3, "keen": 0.8, "upfront": 1.2, "forthcoming": 1, "communicative": 1.5,
    "personable": 1.5, "personal": 0.5, "friendliest": 2, "nicest": 2, "kindest": 2, "mates": 0.5,
    "dealt": 0.2, "obliged": 0.5, "decent": 0.6,
}

NEG_WORDS = {
    # strong
    "terrible": 2, "awful": 2, "appalling": 2, "disgraceful": 2, "disgrace": 2, "disgusting": 2,
    "horrible": 2, "horrendous": 2, "dreadful": 2, "pathetic": 2, "useless": 2, "hopeless": 2,
    "shocking": 2, "atrocious": 2, "unacceptable": 2, "ridiculous": 1.8, "outrageous": 2,
    "incompetent": 2, "incompetence": 2, "joke": 1.5, "rip-off": 2, "ripoff": 2, "scam": 2,
    "worst": 2, "rude": 2, "rudeness": 2, "arrogant": 2, "dismissive": 1.8, "disrespectful": 2,
    "unprofessional": 2, "fuming": 2, "furious": 2, "livid": 2, "disgusted": 2, "appalled": 2,
    "dishonest": 2, "lied": 2, "lying": 2, "liar": 2, "misled": 1.8, "deceptive": 2, "ripped": 1.5,
    "negligent": 2, "negligence": 2, "damaged": 1.5, "damage": 1.2, "failure": 1.5, "disaster": 2,
    "nightmare": 2, "shambles": 2, "farce": 2, "debacle": 2, "abysmal": 2, "woeful": 2, "dire": 1.8,
    "sloppy": 1.8, "shoddy": 1.8, "botched": 2, "mess": 1.5, "messy": 1.2, "smirk": 1.5,
    # moderate
    "bad": 1.5, "poor": 1.5, "poorly": 1.5, "disappointed": 1.5, "disappointing": 1.5,
    "disappointment": 1.5, "unhappy": 1.5, "annoyed": 1.5, "annoying": 1.5, "frustrated": 1.5,
    "frustrating": 1.5, "frustration": 1.5, "angry": 1.8, "upset": 1.5, "irritated": 1.5,
    "unimpressed": 1.5, "dissatisfied": 1.5, "wasted": 1.5, "waste": 1.5, "hassle": 1,
    "wait": 0.4, "waited": 1, "waiting": 0.8, "ages": 1, "forever": 1.2, "slow": 1.2, "slowly": 1,
    "late": 1, "delayed": 1.2, "delay": 1, "delays": 1, "overcharged": 2, "overcharge": 1.5,
    "overpriced": 1.8, "expensive": 1.2, "pricey": 1.2, "dearer": 1, "higher": 0.8, "gouging": 2,
    "chase": 1.2, "chased": 1, "chasing": 1.2, "ignored": 1.8, "ignoring": 1.5, "unanswered": 1.5,
    "silence": 1.2, "silent": 1, "unresponsive": 1.8, "uncontactable": 1.8, "mistake": 1.2,
    "mistakes": 1.2, "wrong": 1.2, "error": 1, "errors": 1, "faulty": 1.5, "broken": 1.5, "broke": 1.2,
    "scratch": 1.2, "scratched": 1.5, "scuff": 1, "scuffed": 1.2, "dent": 1.2, "dented": 1.5,
    "dirty": 1.5, "filthy": 2, "grease": 0.8, "greasy": 1.2, "stained": 1.2, "stain": 1,
    "leak": 1.2, "leaking": 1.2, "leaks": 1.2, "rattle": 1, "rattling": 1.2, "rattles": 1,
    "noise": 0.6, "noisy": 0.8, "loose": 1, "missing": 1.2, "lost": 1.2, "forgot": 1.5, "forgotten": 1.5,
    "unfinished": 1.5, "incomplete": 1.5, "unclear": 1.2, "confusing": 1.2, "confused": 1,
    "misleading": 1.5, "vague": 1, "stressful": 1.2, "stressed": 1, "stress": 0.8, "worried": 1,
    "worry": 0.5, "concerned": 1, "concern": 0.6, "concerns": 0.6, "complaint": 1.2, "complain": 1.2,
    "complained": 1.2, "complaints": 1.2, "complaining": 1.2, "problem": 0.8, "problems": 0.8,
    "issue": 0.6, "issues": 0.6, "fault": 1, "trouble": 1, "inconvenience": 1.2,
    "inconvenient": 1.2, "inconvenienced": 1.2, "unreliable": 1.8, "inconsistent": 1.2,
    "careless": 1.8, "lazy": 1.8, "grumpy": 1.5, "snappy": 1.2, "sharp": 0.8, "abrupt": 1.2,
    "curt": 1.5, "short": 0.3, "cold": 0.8, "unfriendly": 1.8, "unhelpful": 1.8, "uninterested": 1.5,
    "indifferent": 1.5, "condescending": 2, "patronising": 2, "patronizing": 2, "eye-roll": 1.5,
    "eyeroll": 1.5, "eyes": 0, "sigh": 0.8, "sighed": 0.8, "huffed": 1.2, "snapped": 1.5,
    "blamed": 1.5, "blame": 1.2, "excuses": 1.5, "excuse": 1, "runaround": 1.8, "run-around": 1.8,
    "pushy": 1.8, "pressured": 1.5, "pressure": 1, "hounded": 1.8, "harassed": 2, "badgered": 1.8,
    "hidden": 1.2, "unexpected": 0.8, "unauthorised": 1.8, "unauthorized": 1.8, "unnecessary": 1.2,
    "bogus": 1.8, "padded": 1.5, "inflated": 1.5, "jacked": 1.2, "extortionate": 2, "steep": 1,
    "ruined": 2, "unsafe": 2, "dangerous": 2, "worse": 1.5, "worsened": 1.5, "regret": 1.5,
    "regretted": 1.5, "avoid": 1.2, "never": 0, "unsatisfactory": 1.8, "subpar": 1.5,
    "lacking": 1.2, "lack": 1, "disorganised": 1.8, "disorganized": 1.8, "chaotic": 1.8,
    "chaos": 1.8, "cluttered": 1, "cramped": 0.8, "uncomfortable": 1, "stuffy": 0.8,
    "unwelcome": 1.5, "unwelcoming": 1.8, "dead": 0, "cancelled": 1, "canceled": 1,
    "rescheduled": 0.8, "bumped": 1, "stuck": 0.8, "stranded": 1.5, "overdue": 1.2,
    "unresolved": 1.5, "unaddressed": 1.5, "uncaring": 1.8, "fobbed": 1.8, "brushed": 1,
    "ghosted": 2, "nobody": 0.8, "slack": 1.2, "hold": 0.3, "queue": 0.3, "queued": 0.5,
    "bothered": 0.8, "tired": 0.8, "fed": 0.3, "sick": 0.8, "mislaid": 1.2, "misplaced": 1.2,
    "mishandled": 1.8, "misdiagnosed": 1.8, "overbooked": 1.5, "understaffed": 1.2, "unprepared": 1.5,
    "untidy": 1.2, "smelly": 1.2, "smell": 0.5, "smelt": 0.8, "wet": 0.6, "sticky": 0.8,
    "cheap": 0.3, "cheapest": 0.3, "tacky": 1, "dismay": 1.5, "dismayed": 1.5, "shame": 1,
    "sad": 1, "sadly": 1, "unfortunately": 1, "wish": 0.6, "hoping": 0.3, "insulting": 2,
    "insulted": 2, "belittled": 2, "talked-down": 2, "overlooked": 1.2, "unreasonable": 1.5,
    "nonsense": 1.5, "rubbish": 1.8, "garbage": 1.8, "crap": 1.8, "crappy": 1.8, "sucks": 1.8,
    "stupid": 1.8, "idiot": 2, "idiots": 2, "clueless": 1.8, "inept": 1.8, "bungled": 1.8,
}

# Phrases are matched BEFORE single words. Weight can be positive or negative.
PHRASES = {
    # --- positive phrases
    "no dramas": 1.5, "no worries": 1.2, "no drama": 1.5, "no hassle": 1.5, "no fuss": 1.5,
    "no stress": 1.5, "no problem": 1, "no problems": 1, "no issues": 1, "no complaints": 1.5,
    "no surprises": 1.5, "no hidden": 1.5, "no pressure": 1.5, "no pushy": 1.5, "no hard sell": 1.5,
    "couldn't be easier": 2, "couldn't have been easier": 2, "couldn't fault": 2, "can't fault": 2,
    "could not fault": 2, "cannot fault": 2, "wouldn't change a thing": 2, "would not change a thing": 2,
    "not a single complaint": 2, "not one complaint": 2, "went above and beyond": 2,
    "above and beyond": 2, "went out of their way": 2, "went out of his way": 2,
    "went out of her way": 2, "out of their way": 1.5, "out of his way": 1.5, "out of her way": 1.5,
    "kept me posted": 2, "kept me informed": 2, "kept me in the loop": 2, "kept in the loop": 2,
    "kept us posted": 2, "kept us informed": 2, "keep me posted": 1, "in the loop": 1,
    "heads up": 1, "ready before": 1.5, "ready early": 1.5, "ready ahead": 1.5, "back early": 1.5,
    "well before the time": 1.5, "ahead of schedule": 1.5, "ahead of time": 1, "on time": 1.2,
    "on schedule": 1.2, "right on time": 1.5, "under budget": 1.8, "under quote": 1.8,
    "less than i expected": 1.5, "less than expected": 1.5, "cheaper than i": 1.5,
    "cheaper than expected": 1.5, "lower than i": 1.2, "less than quoted": 1.5,
    "less than the quote": 1.5, "within budget": 1.2, "on budget": 1, "good value": 1.5,
    "great value": 2, "value for money": 1.5, "fair price": 1.5, "fair pricing": 1.5,
    "fair prices": 1.5, "reasonably priced": 1.5, "well priced": 1.5, "good price": 1.2,
    "great price": 1.5, "plain english": 1.5, "in plain english": 1.5, "made sense": 1.2,
    "makes sense": 0.8, "easy to follow": 1.5, "easy to understand": 1.5, "easy to book": 1.5,
    "easy to deal with": 1.5, "easy to work with": 1.5, "easy to talk to": 1.5,
    "put my mind at ease": 2, "put me at ease": 2, "put us at ease": 2, "at ease": 1,
    "made us feel": 0.8, "felt welcome": 1.5, "felt welcomed": 1.5,
    "felt valued": 1.5, "felt looked after": 1.5, "felt at home": 1.5, "right at home": 1.5,
    "looked after": 1.2, "took the time": 1.2, "took the time to": 1.2, "went through": 0.5,
    "walked me through": 1.2, "talked me through": 1.2, "talked us through": 1.2,
    "took me through": 1.2, "nothing was too much": 2, "nothing is too much": 2, "too much trouble": -0.5,
    "in a good way": 1.5, "pleasantly surprised": 2, "happily surprised": 2, "pleasant surprise": 2,
    "top notch": 2, "top-notch": 2, "five stars": 2, "5 stars": 2, "10 out of 10": 2, "10/10": 2,
    "highly recommend": 2, "would recommend": 1.5, "will be back": 1.2, "will be returning": 1.2,
    "will return": 1, "will definitely be back": 2, "come back again": 1,
    "second to none": 2, "head and shoulders": 1.5, "a pleasure": 1.5, "pleasure to deal": 2,
    "good on them": 1.5, "good on you": 1.5, "good on ya": 1.5, "well done": 1.5, "good job": 1.5,
    "great job": 1.5, "nice touch": 1.5, "nice work": 1.5, "top marks": 2, "full marks": 2,
    "top bloke": 1.5, "top blokes": 1.5, "top team": 1.5, "top service": 2, "solid job": 1.5,
    "spot on": 1.5, "hit the spot": 1.2, "all the way through": 0.5,
    "from start to finish": 0.8, "start to finish": 0.8, "every step": 0.6, "every step of the way": 1,
    "lightning quick": 2, "lightning fast": 2, "quick as": 1, "quicker than": 1.2, "faster than": 1.2,
    "sooner than": 1.5, "earlier than": 1.5, "well explained": 1.5, "so well": 1, "very well": 1,
    "didn't have to ask": 1.2, "didn't need to chase": 1.8, "didn't have to chase": 1.8,
    "never had to chase": 1.8, "never had to ring": 1.8, "never had to call": 1.8,
    "without having to ask": 1.5, "without asking": 0.3, "didn't have to wait": 1.5,
    "no wait": 1, "no waiting": 1, "no chasing": 1.8, "no need to chase": 1.8, "no need to ring": 1.8,
    "they rang me": 1.2, "they called me": 1.2, "they texted me": 1.2, "they text me": 1.2,
    "rang me back": 1.2, "called me back": 1.2, "called back": 0.8, "rang back": 0.8,
    "got back to me": 1.2, "got back to us": 1.2, "returned my call": 1.5, "replied quickly": 1.5,
    "replied promptly": 1.5, "answered all my": 1.5, "answered every": 1.5, "answered my questions": 1.5,
    "answered our questions": 1.5, "all my questions": 0.8, "every question": 0.8,
    "text updates": 1, "regular updates": 1.5, "constant updates": 1.5, "kept updated": 1.5,
    "updated me": 1.2, "updates along the way": 1.5, "sent me updates": 1.5, "sending me updates": 1.5,
    "kept sending me updates": 1.5, "sent photos": 1, "sent me photos": 1, "sent pictures": 1,
    "showed me": 1, "showed us": 1, "drew me": 1.2,
    "dead easy": 1.5, "dead set legend": 2, "bloody legend": 2, "absolute legend": 2,
    "absolute pleasure": 2, "absolutely wonderful": 2, "work of art": 1.5, "clean and tidy": 1.5,
    "clean as": 1, "spotless": 1.5, "cleaned the car": 1, "washed and vacuumed": 1.5, "vacuumed": 1,
    "free wash": 1, "complimentary": 1, "loan car": 0.5, "courtesy car": 0.5, "free of charge": 1,
    "no charge": 1, "didn't charge": 1.2, "didn't charge me": 1.5, "waived": 1.2, "discount": 0.8,
    "discounted": 0.8, "threw in": 1.2, "chucked in": 1.2, "extra mile": 1.8, "extra effort": 1.5,
    "braced for": 0.3, "i was dreading": 0.3, "was nervous": 0.3,
    "so easy": 1.5, "very easy": 1.5, "really easy": 1.5, "super easy": 1.5, "made it easy": 1.8,
    "made it so easy": 2, "made things easy": 1.8, "stress free": 1.5, "hassle free": 1.5,
    "worry free": 1.5, "all good": 1.2, "all sorted": 1.5, "sorted quickly": 1.5,
    "done and dusted": 1.2, "done in": 0.3, "same day": 1, "next day": 0.3,
    "while i waited": 0.8, "within the hour": 1.2, "within an hour": 1.2, "ready when": 1.2,
    "ready on time": 1.5, "ready by": 1, "ready for pickup": 0.5, "ready to go": 0.8,
    "went smoothly": 1.8, "ran smoothly": 1.8, "went well": 1.5, "went really well": 2,
    "worked out well": 1.5, "turned out well": 1.5, "happy with": 1.5, "pleased with": 1.5,
    "satisfied with": 1.2, "impressed with": 1.5, "thrilled with": 2, "stoked with": 2,
    "love the": 1.2, "loved the": 1.2, "love my": 1.2, "love how": 1,
    "really appreciate": 1.5, "really appreciated": 1.5, "much appreciated": 1.5,
    "thank you": 1, "thanks heaps": 1.2, "thanks so much": 1.5, "many thanks": 1.2,
    "big thanks": 1.2, "huge thanks": 1.5, "thanks a lot": 1, "thanks again": 1,
    "had a chat": 0.5, "friendly chat": 1, "great chat": 1.5, "good chat": 1,
    "say hello": 0.8, "said hello": 0.8, "said hi": 0.8, "greeted me": 1, "greeted us": 1,
    "knew my name": 1.5, "remembered my name": 1.5, "remembered me": 1.5, "recognised me": 1.2,
    "not rushed": 1.5, "wasn't rushed": 1.5, "wasn't pushy": 1.8, "wasn't a bother": 1.5,
    "not pushy": 1.8, "never pushy": 1.8, "never felt pressured": 1.8, "didn't feel pressured": 1.8,
    "not once": 0.3, "not a single": 0.3, "not one eye-roll": 1.5, "not one eye roll": 1.5,
    "not the fake kind": 1.5, "not fake": 1.5, "real thing": 1.2, "the real deal": 1.5,
    "first class": 2, "world class": 2, "five star": 2, "5 star": 2,
    "above expectations": 2, "exceeded expectations": 2, "exceeded my expectations": 2,
    "better than expected": 1.8, "better than i expected": 1.8, "better than i thought": 1.8,
    "cheaper than i thought": 1.8, "heaps cheaper": 1.8, "way cheaper": 1.8, "much cheaper": 1.8,
    "much better": 1.5, "so much better": 1.8, "a lot better": 1.5, "far better": 1.5,
    "miles better": 1.8, "night and day": 1, "breath of fresh air": 2, "refreshing change": 1.5,
    "makes a real difference": 1.5, "real difference": 1,

    # --- negative phrases (service failures typical for this dataset)
    "had to chase": -1.8, "have to chase": -1.8, "having to chase": -1.8, "had to ring": -1.5,
    "have to ring": -1.5, "having to ring": -1.5, "had to call": -1.2, "have to call": -1.2,
    "having to call": -1.2, "had to follow up": -1.5, "had to ask": -1, "had to keep asking": -1.8,
    "had to keep ringing": -1.8, "had to keep calling": -1.8, "had to phone": -1.3,
    "had to email": -1.2, "had to go in": -1, "had to come back": -1.5, "had to return": -1.2,
    "had to wait": -1.2, "had to leave": -0.8, "had to pay": -0.8, "had to drive": -0.5,
    "had to take": -0.5, "had to demand": -2, "had to push": -1.5, "had to insist": -1.5,
    "had to explain": -1.2, "had to remind": -1.5, "had to point out": -1.5, "have to keep": -1.2,
    "chase up": -1.5, "chased up": -1.5, "chase them": -1.8, "chase you": -1.8, "chase for": -1.8,
    "chase chase": -2, "chasing them": -1.8, "chasing up": -1.8, "chasing for": -1.8,
    "phone tag": -1.8, "radio silence": -2, "black hole": -1.8, "no response": -1.8,
    "no reply": -1.8, "no answer": -1.5, "no call": -1.5, "no calls": -1.5, "no text": -1.5,
    "no texts": -1.5, "no email": -1.5, "no updates": -1.8, "no update": -1.8, "no contact": -1.8,
    "no communication": -2, "no explanation": -1.8, "no apology": -1.8, "no one rang": -2,
    "no one called": -2, "no one got back": -2, "no one told": -1.8, "no one explained": -1.8,
    "no one let": -1.8, "no one contacted": -2, "no one replied": -2, "no one answered": -2,
    "no one bothered": -2, "no one was": -1, "no one knew": -1.8, "no one seemed": -1.5,
    "no record": -1.5, "no sign": -1.2, "no notice": -1.5, "no warning": -1.5,
    "no heads up": -1.8, "no courtesy": -1.8, "no loan car": -1.5, "no courtesy car": -1.5,
    "nobody rang": -2, "nobody called": -2, "nobody told": -1.8, "nobody let": -1.8,
    "nobody got back": -2, "nobody contacted": -2, "nobody replied": -2, "nobody answered": -2,
    "nobody bothered": -2, "nobody knew": -1.8, "nobody explained": -1.8, "nobody seemed": -1.5,
    "nobody was": -1, "nobody has": -1.5, "nobody came": -1.5, "nobody even": -1.8,
    "nothing since": -1.8, "nothing from": -1.2, "heard nothing": -1.8, "hear nothing": -1.8,
    "hearing nothing": -1.8, "heard absolutely nothing": -2, "not a word": -1.8, "not a peep": -1.8,
    "not heard": -1.5, "never heard": -1.5, "never called": -2, "never rang": -2, "never got back": -2,
    "never replied": -2, "never responded": -2, "never contacted": -2, "never arrived": -1.8,
    "never came": -1.5, "never received": -1.8, "never told": -1.8, "never explained": -1.8,
    "never let me know": -2, "never lets me know": -2, "never informed": -1.8, "never followed": -1.8,
    "never fixed": -2, "never done": -1.5, "never happened": -1.5, "never showed": -1.5,
    "never hears": -1.8, "never heard back": -2, "never got": -1.5, "never gets": -1.2,
    "never answer": -1.8, "never answered": -1.8, "never bothered": -2, "never got a": -1.5,
    "zero contact": -2, "zero communication": -2, "zero updates": -2, "zero calls": -2,
    "zero call": -2, "zero updates": -2, "zero explanation": -2, "zero follow": -2, "zero effort": -2,
    "zero interest": -2, "zero help": -2, "zero response": -2, "zero reply": -2, "zero notice": -2,
    "calls though, zero": -2, "calls, zero": -2, "zero.": -0.8, "non-existent": -1.8,
    "nonexistent": -1.8, "non existent": -1.8, "left on hold": -1.5, "on hold for": -1.3,
    "on hold for ages": -2, "hold music": -1, "kept me waiting": -1.8, "kept us waiting": -1.8,
    "kept waiting": -1.8, "left waiting": -1.8, "left me waiting": -1.8, "left us waiting": -1.8,
    "waited ages": -2, "wait for ages": -2, "waiting for ages": -2, "waited for ages": -2,
    "waited around": -1.5, "stood at": -0.8,
    "stood around": -1.2, "standing around": -1.2, "took forever": -2, "taking forever": -2,
    "took ages": -2, "taking ages": -2, "took so long": -1.8, "taking so long": -1.8,
    "took way too long": -2, "took too long": -1.8, "too long": -1.5, "so long": -0.5,
    "long wait": -1.8, "long time": -1, "whole day": -1.2, "whole saturday": -1.2,
    "entire day": -1.2, "half the day": -1.2, "wasted my": -2, "wasted a": -1.8, "wasted the": -1.8,
    "waste of": -1.8,  "waste of time": -2, "waste of money": -2,
    "not ready": -1.5, "wasn't ready": -1.5, "weren't ready": -1.5, "still not ready": -2,
    "still waiting": -1.8, "still no": -1.8, "still hasn't": -1.8, "still haven't": -1.8,
    "still has not": -1.8, "still have not": -1.8, "still not": -1.5, "still isn't": -1.5,
    "still doesn't": -1.5, "still wasn't": -1.5, "still the same": -1.5, "still there": -1.2,
    "still rattling": -2, "still leaking": -2, "still making": -1.8, "still broken": -2,
    "same problem": -1.8, "same issue": -1.8, "same noise": -1.8, "same fault": -1.8,
    "problem persists": -1.8, "not fixed": -2, "wasn't fixed": -2, "not repaired": -2,
    "wasn't repaired": -2, "not resolved": -2, "wasn't resolved": -2, "not sorted": -1.8,
    "wasn't sorted": -1.8, "not done": -1.5, "wasn't done": -1.5, "not fixed properly": -2,
    "second time": -1.2, "third time": -1.8, "fourth time": -2, "fifth time": -2, "again and again": -1.5,
    "second visit": -1.2, "third visit": -1.8, "three visits": -1.8, "two visits": -1.2,
    "four visits": -2, "multiple visits": -1.8, "multiple times": -1.2, "several times": -1.2,
    "three times": -1.2, "four times": -1.5, "five times": -1.8, 
    "keep being told": -1.8, "kept being told": -1.8, "kept telling me": -1.5,
    "kept saying": -1.2, "keeps saying": -1.2, "keep saying": -1.2, "told me it would": -0.5,
    "was promised": -1.2, "were promised": -1.2, "promised me": -1, "promised a call": -1.8,
    "promised to call": -1.8, "promised to ring": -1.8, "said they'd call": -1.2, "said they would call": -1.2,
    "said they'd ring": -1.2, "said they would ring": -1.2, "would call me": -0.8, "would ring me": -0.8,
    "should have": -1.2, "should've": -1.2, "shouldn't have": -1.2,
    "wish they": -1, "wish i": -0.8, "wish the": -0.8,
    "wish someone": -1, "would love to know": -1, "would like to know": -0.6, 
    "didn't call": -1.8, "didn't ring": -1.8, "didn't text": -1.5, "didn't email": -1.5,
    "didn't tell": -1.5, "didn't let me know": -2, "didn't let us know": -2, "didn't inform": -1.8,
    "didn't explain": -1.8, "didn't listen": -2, "didn't bother": -1.8, "didn't answer": -1.8,
    "didn't reply": -1.8, "didn't respond": -1.8, "didn't get back": -1.8, "didn't follow": -1.5,
    "didn't fix": -2, "didn't arrive": -1.8, "didn't show": -1.5, "didn't even": -1.5,
    "didn't seem": -0.8, "didn't know": -1.2, "didn't have": -0.8, "didn't offer": -1.2,
    "didn't apologise": -1.8, "didn't apologize": -1.8, "didn't care": -2, "didn't ask": -1.2,
    "didn't authorise": -2, "didn't authorize": -2, "didn't approve": -2, "didn't agree": -1.2,
    "wasn't asked": -1.5, "wasn't told": -1.8, "wasn't informed": -1.8, "wasn't offered": -1.2,
    "wasn't given": -1.2, "wasn't contacted": -1.8, "wasn't called": -1.8, "wasn't happy": -1.8,
    "wasn't impressed": -1.8, "wasn't good": -1.8, "wasn't clear": -1.5, "wasn't there": -1.2,
    "wasn't even": -1.2, "wasn't what": -1.2, "weren't told": -1.8, "weren't informed": -1.8,
    "weren't happy": -1.8, "hasn't been": -1, "hasn't arrived": -1.8, "hasn't called": -1.8,
    "hasn't rung": -1.8, "hasn't replied": -1.8, "hasn't responded": -1.8, "hasn't got": -1,
    "hasn't come": -1.5, "hasn't happened": -1.5, "hasn't heard": -1.5, "hadn't arrived": -1.8,
    "hadn't called": -1.8, "hadn't rung": -1.8, "hadn't been": -1, "hadn't even": -1.5,
    "hadn't started": -1.8, "hadn't done": -1.8, "hadn't touched": -1.8, "haven't heard": -1.8,
    "haven't had": -1.2, "haven't been": -1.2, "haven't received": -1.8, "haven't called": -1.8,
    "haven't rung": -1.8, "haven't spoken": -1.5, "haven't met": -1.2, "haven't got": -1.2,
    "haven't seen": -1.2, "haven't even": -1.5, "haven't replied": -1.8, "haven't responded": -1.8,
    "can't get": -1.2, "can't reach": -1.8, "couldn't get": -1.5, "couldn't reach": -1.8,
    "couldn't find": -1.5, "couldn't tell": -1.2, "couldn't give": -1.5, "couldn't say": -1.2,
    "couldn't explain": -1.8, "couldn't understand": -1.5, "couldn't answer": -1.8,
    "can't understand": -1.2, "cannot get": -1.2, "unable to": -1.2, "wouldn't listen": -2,
    "wouldn't let": -1.5, "wouldn't take": -1.5, "wouldn't budge": -2, "wouldn't give": -1.8,
    "wouldn't answer": -1.8, "wouldn't explain": -1.8, "wouldn't tell": -1.8, "wouldn't honour": -2,
    "won't answer": -1.5, "won't listen": -2, "won't go down": -1.5, "won't go up": -1.5,
    "won't start": -2, "won't work": -2, "won't close": -1.5, "won't open": -1.5, "won't lock": -1.5,
    "doesn't work": -2, "doesn't lock": -1.5, "doesn't close": -1.5, "doesn't open": -1.5,
    "doesn't start": -2, "doesn't listen": -2, "doesn't care": -2, "don't care": -2, "don't listen": -2,
    "don't know": -1, "don't understand": -1.2, "do not understand": -1.2,
    "not working": -1.8, "not listen": -2, "not listened": -2, "not listening": -2, "not heard": -1.5,
    "not told": -1.8, "not informed": -1.8, "not contacted": -1.8, "not called": -1.8,
    "not answered": -1.8, "not replied": -1.8, "not returned": -1.8, "not given": -1.2,
    "not asked": -1.5, "not offered": -1.2, "not explained": -1.8, "not authorised": -2,
    "not authorized": -2, "not approved": -2, "not agreed": -1.5, "not there": -1.2, "not happy": -1.8,
    "not impressed": -1.8, "not good": -1.8, "not great": -1.5, "not cool": -1.5, "not acceptable": -2,
    "not satisfied": -1.8, "not pleased": -1.8, "not what": -1.2, "not worth": -1.8,
    "not right": -1.5, "not clear": -1.5, "not helpful": -1.8, "not friendly": -1.8,
    "not professional": -1.8, "not interested": -1.5, 
    "not back": -1,  "not here": -1.2, "not available": -1.2, "not in stock": -1.5,
    "out of stock": -1.5, "back order": -1.2, "backorder": -1.2, "back-ordered": -1.2,
     "hadn't arrived yet": -1.8, "hasn't arrived yet": -1.8, "still waiting for": -1.8,
    "waiting on": -0.8, "waiting for": -0.8, "waiting to hear": -1.5, "wondering whether": -1.5,
    "wondering if": -1.2, "wondering when": -1.2, "wondering what": -1.2, "no idea what": -1.2,
    "no idea when": -1.5, "no idea if": -1.5, "no idea whether": -1.5, "in the dark": -1.8,
    "left in the dark": -2, "kept in the dark": -2, "not knowing": -1.5,
    "not sure what": -0.8, "not sure when": -1, "not sure if": -0.8,
    "went straight to": -0.5, "went to voicemail": -1.8, 
    "left messages": -1.2, "left several messages": -1.8, 
    "left me a message": 0.3, "rang and rang": -1.8, "tried calling": -1.2, "tried ringing": -1.2,
    "tried to call": -1.2, "tried to ring": -1.2, "tried to contact": -1.2, "tried to get": -1,
    "tried to book": -0.8, "tried to explain": -0.5, "tried to sell": -1.8, "tried to upsell": -2,
    "tried to push": -1.8, "tried to charge": -2, "trying to sell": -1.5, "upsell": -1.8,
    "up-sell": -1.8, "add-ons": -1, "add ons": -1, "add-on": -1, "extras i didn't": -2,
    "didn't ask for": -1.8, "didn't request": -1.8, "didn't order": -1.8, "didn't book": -1.8,
    "didn't need": -1.5, "never asked for": -2, "never ordered": -2, "never booked": -2,
    "never agreed": -2, "never authorised": -2, "never authorized": -2, "never approved": -2,
    "never on the quote": -2, "wasn't on the quote": -2, "not on the quote": -2, "not on the job card": -2,
    "on top of the quote": -1.5, "more than quoted": -2, "more than the quote": -2, "more than i": -0.8,
    "more than expected": -1.5, "higher than": -1.2, "bigger than the quote": -2, "bigger than quoted": -2,
    "above the quote": -1.8, "above quote": -1.8, "over the quote": -1.8, "over quote": -1.8,
    "over quoted": -1.8, "over the quoted": -1.8, "went up": -0.8, "went over": -1.2, "blew out": -1.8,
    "blown out": -1.8, "grew by": -1.5, "added to the bill": -1.8, "added to my bill": -1.8,
    "added on": -1.2, "snuck in": -1.8, "sneaked in": -1.8, "slipped in": -1.5, "tacked on": -1.8,
    "just added": -1.5, "was added": -1.2, "were added": -1.2, "appeared on": -1.5, "appeared on my bill": -2,
    "adds up": -1, "add up": -1, "nickel and dime": -2, "nickel-and-dime": -2, "ripped off": -2,
    "rip off": -2, "rip-off": -2, "too expensive": -1.8, "too much": -1.2, "too high": -1.5,
    "way too": -1.2, "far too": -1.2, "so expensive": -1.8, "very expensive": -1.8, "bit steep": -1.2,
    "pretty steep": -1.5, "a bit much": -1.2, "bit much": -1.2, "bit pricey": -1.2, "bit dear": -1.2,
    "not cheap": -1.2, "not impressed with": -1.8, "not happy with": -1.8, "not pleased with": -1.8,
    "unhappy with": -1.8, "disappointed with": -1.8, "disappointed in": -1.8, "annoyed with": -1.5,
    "annoyed that": -1.5, "frustrated with": -1.5, "frustrated that": -1.5, "upset with": -1.5,
    "upset that": -1.5, "fed up": -2, "sick of": -1.8, "tired of": -1.8, "over it": -1.5,
    "gave up": -1.5, "give up": -1.5, "given up": -1.5, "giving up": -1.5, 
     "drove in instead": -1, "went elsewhere": -2, "take my business": -2,
    "taking my business": -2, "take our business": -2, "never again": -2, "won't be back": -2,
    "will not be back": -2, "won't be returning": -2, "won't return": -2, "not returning": -2,
    "last time": -1.5, "last visit": -1, "will go elsewhere": -2, "going elsewhere": -2,
    "look elsewhere": -1.8, "looking elsewhere": -1.8, "find another": -1.8, "try another": -1.5,
    "other dealer": -1, "another dealer": -1, "other dealership": -1, "another dealership": -1,
    "avoid this": -2, "stay away": -2, "do not recommend": -2, "don't recommend": -2,
    "wouldn't recommend": -2, "would not recommend": -2, "can't recommend": -2, "not recommend": -2,
    "far from": -1.2, "far from happy": -2, "less than impressed": -1.8, "less than happy": -1.8,
    "less than ideal": -1.5, "less than satisfactory": -1.8, "below par": -1.8, "not up to": -1.5,
    "let down": -2, "let me down": -2, "let us down": -2, "letting me down": -2, 
      "left without": -1.5, "left stranded": -2, "left standing": -1.5,
    "brushed off": -1.8, "fobbed off": -2, "blown off": -1.8, "palmed off": -1.8, "passed around": -1.8,
    "passed from": -1.5, "passed to": -0.8, "pass the buck": -2, "buck passing": -2,
    "talked over": -1.8, "talked down": -2, "spoke down": -2, "looked down": -1.8, "made me feel": 0.3,
    "made me feel stupid": -2, "made me feel small": -2, "made me feel like a": -1.5,
    "made me feel unwelcome": -2, "felt ignored": -2, "felt dismissed": -2, "felt rushed": -1.5,
    "felt pressured": -1.8, "felt like a nuisance": -2, "felt like an inconvenience": -2,
    "felt brushed": -1.8, "felt unwelcome": -2, "felt taken advantage": -2, "taken advantage": -2,
    "took advantage": -2, "eye roll": -1.5, "eye-roll": -1.5, "rolled their eyes": -1.8,
    "rolled her eyes": -1.8, "rolled his eyes": -1.8, "rolling their eyes": -1.8, "got an eye": -1.5,
    "looked at me like": -1.5, "spoke to me like": -1.8, "talked to me like": -1.8, "treated me like": -1,
    "treated like": -1, "treated badly": -2, "treated poorly": -2, "treated terribly": -2,
    "not taken seriously": -2, "wasn't taken seriously": -2, 
     "no interest": -1.8, "no effort": -1.8, "no care": -1.8, "no help": -1.8,
    "no support": -1.8, "no follow up": -1.8, "no follow-up": -1.8, "no accountability": -2,
    "no ownership": -2, "no empathy": -2, "no compassion": -2, "no manners": -2, "no clue what": -1.5,
    "no refund": -1.8, "no discount": -1, "no offer": -1.2, "no sorry": -1.8, "no thanks": -0.2,
    "no one cares": -2, "nobody cares": -2, "nobody cared": -2, "no one cared": -2,
    "stood at the front desk": -1.2, "front desk for": -0.8, "at the counter": -0.3, "at the desk": -0.3,
    "ate my": -1.5, "ate the whole": -1.5, "ate up": -1.2, "eats up": -1.2, "cost me": -1,
    "wasn't there before": -2, "not there before": -2, "weren't there before": -2, "wasn't there when": -2,
    "wasn't there when i": -2, "new scratch": -2, "new dent": -2, "new noise": -2, "new rattle": -2,
    "new mark": -1.8, "new marks": -1.8, "new damage": -2, "grease marks": -1.8, "greasy marks": -1.8,
    "dirty hand": -1.8, "hand marks": -1.5, "handprints": -1.5, "fingerprints": -1, "oil on": -1.2,
    "mud inside": -1.8, "dirt inside": -1.8, "left a mess": -2, "left it dirty": -2, "came back dirty": -2,
    "came back with": -1, "came back worse": -2, "came back still": -1.8, "came back the same": -1.8,
    "came back scratched": -2, "came back damaged": -2, "returned dirty": -2, "returned with": -0.8,
    "returned damaged": -2, "returned scratched": -2, 
    "was damaged": -2, "got damaged": -2, "been damaged": -2, "was scratched": -2, "got scratched": -2,
    "been scratched": -2, "was dented": -2, "got dented": -2, "was dirty": -1.8, "was filthy": -2,
    "was loose": -1.5, "was missing": -1.5, "went missing": -1.8, "is missing": -1.5, "are missing": -1.5,
    "was left on": -0.8, "was left with": -0.8, "was left in": -0.8,
    "was lost": -1.5, "got lost": -1.5, "been lost": -1.5, "was forgotten": -1.8, "was ignored": -2,
    "was dismissive": -2, "was rude": -2, "was unhelpful": -1.8, "was abrupt": -1.5, "was curt": -1.8,
    "was short with": -1.8, "was sharp with": -1.8, "was snappy": -1.8, "was condescending": -2,
    "was arrogant": -2, "was disrespectful": -2, "was blunt": -1.2, "was cold": -1.2, "was unfriendly": -1.8,
      
    "pre-delivery inspection": 0, "not back from": -1.5, "come back tomorrow": -1.8, "come back later": -1.5,
    "come back next": -1.2, "come back on": -0.8, "come back in": -0.6, "back tomorrow": -1.2,
    "bring it back": -1.2, "bring the car back": -1.5, "bring it in again": -1.5, 
    "take it back": -1, "took it back": -1, "took it in again": -1.5, "went back": -0.8,
    
     "no show": -1.5, "no-show": -1.5,
    "doesn't add up": -1.5, "didn't add up": -1.5, "not add up": -1.5, "makes no sense": -1.8,
    "make no sense": -1.8, "made no sense": -1.8, "no sense": -1.2, "makes zero": -1.8, "senseless": -1.8,
    "isn't good enough": -2, "not good enough": -2, "isn't acceptable": -2, "isn't okay": -1.5,
    "isn't right": -1.5, "isn't fair": -1.5, "wasn't fair": -1.5, "not fair": -1.5, "not on": -1.2,
    "isn't on": -1.2, "isn't the first": -1.5, "isn't the first time": -2, "not the first": -1.5,
    "not the first time": -2, "same old": -1.5, "same story": -1.5, "same again": -1.5,
    "same as last time": -1.8, "same as before": -1.5, "yet again": -1.8, "once again": -1,
    "over and over": -1.5, "all over the": -1, "all over again": -1.5,
    "all over the invoice": -1.8, "all over the back seat": -1.8, "all over the place": -1.5,
    "runaround": -1.8, "run around": -1.5, "run-around": -1.8, "going around in circles": -1.8,
    "round in circles": -1.8, "ping-pong": -1.5, "ping pong": -1.5, "bounced around": -1.8,
    "bounced between": -1.8, "from pillar to post": -2, "pillar to post": -2, 
     "different story": -1.2, "different answer": -1.2, "conflicting": -1.5,
    "contradicting": -1.5, "contradictory": -1.5,  
    "lack of": -1.5, "lack of communication": -2, "lack of follow": -2, "lack of care": -2,
    "lack of respect": -2, "poor communication": -2, "bad communication": -2, "terrible communication": -2,
    "poor service": -2, "bad service": -2, "terrible service": -2, "appalling service": -2,
    "awful service": -2, "shocking service": -2, "poor customer service": -2, "bad customer service": -2,
    "poor attitude": -2, "bad attitude": -2, "terrible attitude": -2, "poor experience": -2,
    "bad experience": -2, "terrible experience": -2, "awful experience": -2, "horrible experience": -2,
    "worst experience": -2, "worst service": -2, "worst dealership": -2, "worst ever": -2,
    "customer service is": -0.2, "customer service was": -0.2, "customer service from": -0.2,
    "process failure": -2, "system failure": -2, "failure of": -1.5, "breakdown in": -1.5,
    "breakdown of": -1.5, "break down in": -1.5, "communication breakdown": -2, "failed to": -1.8,
    "fail to": -1.8, "failing to": -1.8, "failed me": -2, "fails to": -1.8, "failure to": -1.8,
    "failed again": -2, "fails again": -2, "failed the": -1.5, "failed on": -1.5, "fell through": -1.8,
    "fell short": -1.8, "falls short": -1.8, "falling short": -1.8, "dropped the ball": -2,
    "drop the ball": -2, "dropping the ball": -2, "ball dropped": -2, "slipped through": -1.8,
    "slipped through the cracks": -2, "through the cracks": -1.8, "fallen through": -1.8,
    "lost my": -1.5, "lost the": -1.2, "lost our": -1.5, "lost track": -1.8, "loses track": -1.8,
    "lost it": -1.2, "lost all": -1.5, "lost time": -1.5, "lost a day": -1.8, "lost half": -1.5,
    "losing track": -1.8, "forgot to": -1.8, "forgot about": -1.8, "forgot my": -1.5, "forgot the": -1.5,
    "forgotten about": -1.8, "forget to": -1.8, "forgetting to": -1.8, "slipped their mind": -1.8,
    "slipped my mind": -1, "not my problem": -2, "not their problem": -2, 
     "your problem": -1.2, "not our problem": -2, "wasn't their fault": -0.3,
    "their fault": -1.5, "not my fault": -1,
    "blame the": -1.5, "blamed the": -1.8, "blamed me": -2, "blaming me": -2, "blamed us": -2,
    "put the blame": -1.8, "put it on": -0.5, "put on me": -1.2, "put on hold": -1.3, "on hold": -1,
    "hung up": -2, "hung up on": -2, "hang up": -1.5, "hanging up": -1.5, "put the phone down": -1.8,
    "slammed the": -2, "slammed down": -2, "hung up on me": -2, "cut me off": -2, "cut off": -1.2,
    "talked over me": -2, "interrupted me": -1.8, "wouldn't let me": -1.8, "didn't let me": -1.8,
    "made me wait": -1.8, "made us wait": -1.8, "make me wait": -1.8, "making me wait": -1.8,
    "made me chase": -2, "made me ring": -2, "made me call": -1.8, "made me come": -1.5,
    "made me pay": -1.5, "made me feel like": -1, "forced me to": -1.8, "forced us to": -1.8,
    "forcing me": -1.8, "pushed me": -1.5, "pushed us": -1.5, "pushing me": -1.5, "pushing us": -1.5,
    "pressured me": -2, "pressuring me": -2, "pressured us": -2, "pressure to": -1.5,
    "pressure me": -2, "pressured into": -2, "talked into": -1.8, 
    "tried to talk me": -1.8, "wouldn't take no": -2, "wouldn't take no for an answer": -2,
    "won't take no": -2, "kept asking": -1.5, "kept pushing": -2, "kept trying": -1.5,
     "kept ringing": -1.5, "kept calling": -1.5, "kept ringing me": -2,
    "kept calling me": -2, "keeps calling": -1.5, "keeps ringing": -1.5, "keeps happening": -1.8,
    "keep happening": -1.8, "keeps going": -1, "keeps coming": -1.2, "keep coming": -1.2,
    "keep ringing": -1.5, "keep calling": -1.5, "keep asking": -1.5, "keeps asking": -1.5,
    "keep getting": -1.2, "keeps getting": -1.2, "keep having": -1.2, "keeps having": -1.2,
    "keep waiting": -1.8, "keeps waiting": -1.8, "keep telling": -1.5, "keeps telling": -1.5,
    "keep being": -1.2, "keeps being": -1.2, "kept getting": -1.2, "kept having": -1.2,
    "kept coming": -1.2, "kept happening": -1.8, "kept breaking": -2, "kept failing": -2,
    "kept dropping": -1.8, "kept forgetting": -2, "kept losing": -2, "kept cancelling": -2,
    "kept rescheduling": -2, "kept moving": -1.5, "kept changing": -1.8, "kept pushing back": -2,
    "kept putting": -1.5, "kept putting off": -2, "putting off": -1.5, "put off": -1.5,
    "put it off": -1.5, "pushed back": -1.5, "pushed it back": -1.5, "pushing back": -1.5,
     "rescheduled": -1, "re-scheduled": -1,
    "reschedule": -0.8, "re-schedule": -0.8, "cancelled on": -2, "cancelled my": -2, "cancelled the": -1.5,
    "cancelled with": -1.8, "cancelled again": -2, "cancelled an": -1.5, "another cancelled": -2,
    "last minute": -1.5, "last-minute": -1.5, "short notice": -1.5, "one hour's notice": -1.8,
    "hour's notice": -1.5, "hours notice": -1.5, "no notice": -1.8, "without notice": -1.8,
    "without warning": -1.8, "without telling": -1.8, "without asking me": -2, "without my": -1.8,
    "without permission": -2, "without consent": -2, "without consulting": -2, "without informing": -2,
    "without contacting": -2, "without calling": -1.8, "without letting": -1.8, 
    "without explanation": -1.8, "without a word": -1.8, "without a call": -1.8, "without apology": -1.8,
    "without an apology": -1.8, "without even": -1.5, "without so much": -1.5, "without a single": -1.5,
}

# Words that, when NEGATED ("no", "never", "didn't", "without"), reveal a service failure
EXPECTED_THINGS = {
    "call", "calls", "called", "ring", "rang", "phone", "phoned", "text", "texts", "texted", "email",
    "emails", "emailed", "message", "messages", "contact", "contacted", "update", "updates", "updated",
    "reply", "replied", "response", "responded", "respond", "answer", "answers", "answered",
    "communication", "explanation", "explained", "explain", "apology", "apologised", "apologized",
    "sorry", "notice", "warning", "callback", "follow-up", "followup", "heads-up", "informed", "inform",
    "told", "tell", "listen", "listened", "listening", "care", "cared", "bother", "bothered",
    "offered", "offer", "arrive", "arrived", "arrives", "show", "showed", "turn", "turned",
    "fix", "fixed", "repair", "repaired", "resolve", "resolved", "start", "started", "done",
    "finished", "ready", "help", "helped", "helpful", "assist", "assisted", "assistance",
    "interest", "interested", "effort", "ownership", "accountability", "empathy", "courtesy",
    "record", "idea", "clue", "word", "peep", "sign", "support", "refund", "discount", "loan",
    "courtesy", "spoken", "met", "seen", "received", "returned", "heard", "hear", "hearing",
    "known", "know", "knew", "recognised", "recognized", "touched", "checked", "check",
}

NEGATORS = {
    "no", "not", "never", "nothing", "nobody", "none", "without", "neither", "nor", "zero", "hardly",
    "barely", "cannot", "cant", "wont", "didnt", "dont", "doesnt", "wasnt", "werent", "isnt", "arent",
    "hasnt", "havent", "hadnt", "couldnt", "wouldnt", "shouldnt", "aint", "nowhere", "nope",
}
NEGATION_WINDOW = 3
# words that may sit between a negator and the word it negates without breaking negation scope
NEG_PASSTHROUGH = {"a", "an", "the", "single", "one", "even", "really", "very", "at", "all", "of", "any", "much",
                   "real", "bit", "too", "that", "so", "ever", "once", "particularly", "especially",
                   "overly", "quite", "exactly", "always", "anything", "everything"}

INTENSIFIERS = {"very": 1.3, "really": 1.3, "so": 1.2, "extremely": 1.5, "incredibly": 1.5,
                "absolutely": 1.5, "totally": 1.3, "completely": 1.3, "utterly": 1.5, "truly": 1.3,
                "super": 1.3, "bloody": 1.3, "damn": 1.3, "highly": 1.3, "most": 1.1, "real": 1.1,
                "especially": 1.2, "particularly": 1.2, "genuinely": 1.2, "seriously": 1.3,
                "unbelievably": 1.5, "ridiculously": 1.4, "insanely": 1.4, "beyond": 1.2, "all": 1.1}
DIMINISHERS = {"slightly": 0.5, "somewhat": 0.6, "bit": 0.6, "little": 0.6, "kinda": 0.6, "kind": 0.9,
               "fairly": 0.8, "pretty": 0.9, "rather": 0.8, "mildly": 0.5, "sort": 0.7, "reasonably": 0.8}

# Neutral / lukewarm wording -> tends to Okay
NEUTRAL_PHRASES = [
    "okay", "ok", "alright", "all right", "average", "so-so", "so so", "fine", "acceptable", "adequate",
    "passable", "mediocre", "nothing special", "nothing to write home about", "not bad", "not too bad",
    "could be better", "could have been better", "room for improvement", "middle of the road", "meh",
    "ordinary", "standard", "as expected", "as expected", "expected", "satisfactory", "tolerable",
    "bearable", "livable", "liveable", "neither good nor bad", "it was ok", "it was okay", "fair enough",
    "can't complain", "cant complain", "nothing wrong", "no real issues", "did the job", "does the job",
    "got the job done", "job done", "worked out", "mixed", "hit and miss", "hit or miss", "up and down",
    "not the best", "not the worst", "could do better", "could improve", "decent enough", "good enough",
    "serviceable", "run of the mill", "run-of-the-mill", "nothing more", "nothing less",
]

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "except", "unfortunately", "otherwise",
                  "shame", "only", "just", "still", "whereas", "despite", "sadly", "downside", "negative",
                  "drawback", "catch", "trouble", "apart", "aside"}

# Sarcasm cues: positive word used ironically. Applied per sentence.
SARCASM_CUES = [
    r"\bcheers for (that|nothing|letting|the|your|this|being|keeping|all)\b",
    r"\bthanks for (nothing|the|letting|keeping|calling|ringing|that|looking|your)\b",
    r"\bthank you for (nothing|the|letting|keeping|calling|ringing|that|looking)\b",
    r"\b(five|5) stars? for\b", r"\bwell done\b.*\b(not|never|nothing|still)\b",
    r"\bgreat (job|work|timing|communication|customer service|service|use|start)\b.*\b(guys|team|again|as always|lol|not)\b",
    r"\b(real|really|very|so|truly) (transparent|helpful|great|nice|lovely|good|professional|impressive)\b.*\b(there|guys)\b",
    r"\breally keeping me in the loop\b", r"\breal transparent\b", r"\bgreat value\b.*\b(oh|not)\b",
    r"\bhow (lovely|nice|wonderful|convenient|thoughtful|kind|helpful|professional|efficient)\b",
    r"\blove (how|that|being|rearranging|guessing|waiting|chasing|paying|wondering)\b",
    r"\bjust love\b", r"\breally (love|enjoyed|enjoy|appreciate) (the |being |rearranging|guessing|wondering|waiting|chasing)",
    r"\bso much for\b", r"\bgood (one|job|going)\b.*\b(not|never|nothing|still|again)\b",
    r"\boh (brilliant|great|fantastic|wonderful|lovely|perfect|joy|good|nice|splendid|super|wow)\b",
    r"\b(ha|wow|yeah|oh),? (brilliant|great|fantastic|wonderful|lovely|perfect|thanks|nice|right)\b",
    r"\byeah,? (thanks|right|great|brilliant|sure)\b",
    r"\bfor silence\b", r"\bfor mystery\b", r"\bfor nothing\b", r"\bfor that\.?$",
    r"\bat its finest\b", r"\btruly gripping\b", r"\bthrilling mystery\b", r"\breal(ly)? good at that\b",
    r"\bkeeps life interesting\b", r"\bnot the best\b.*\blol\b", r"\blovely touch\b", r"\bgreat value\b\.?$",
    r"\bwonderful service if you enjoy\b", r"\bbrilliant system\b", r"\bwonderful job,? truly\b",
    r"\bfantastic use of my time\b", r"\bcredit where due\b",
]

# Words that, standing alone as an exclamation, are often sarcastic if negative content is nearby
SARCASM_SINGLES = {"lovely", "brilliant", "great", "fantastic", "wonderful", "perfect", "superb", "terrific",
                   "nice", "cheers", "thanks", "wow", "awesome", "excellent", "amazing", "beautiful",
                   "good", "super", "splendid", "marvellous", "magnificent"}

# Typos seen in this data set -> canonical spelling
TYPO_FIXES = {
    "bying": "buying", "abut": "about", "stck": "stuck", "smoth": "smooth", "somoth": "smooth",
    "plin": "plain", "anotehr": "another", "stlil": "still", "widow": "window", "thier": "their",
    "teh": "the", "recieved": "received", "definately": "definitely", "excelent": "excellent",
    "freindly": "friendly", "frendly": "friendly", "helpfull": "helpful", "wonderfull": "wonderful",
    "terible": "terrible", "horible": "horrible", "disapointed": "disappointed", "dissapointed": "disappointed",
    "dissapointing": "disappointing", "disapointing": "disappointing", "unproffesional": "unprofessional",
    "proffesional": "professional", "profesional": "professional", "curtious": "courteous",
    "curteous": "courteous", "couteous": "courteous", "polight": "polite", "poilte": "polite",
    "grat": "great", "gret": "great", "grate": "great", "gud": "good", "goood": "good", "gd": "good",
    "nce": "nice", "nicee": "nice", "amzing": "amazing", "amazin": "amazing", "awsome": "awesome",
    "fantstic": "fantastic", "fanstastic": "fantastic", "fantasic": "fantastic", "brillant": "brilliant",
    "briliant": "brilliant", "brillent": "brilliant", "briliiant": "brilliant", "lovley": "lovely",
    "lovly": "lovely", "luvly": "lovely", "wnderful": "wonderful", "wondeful": "wonderful",
    "wonderfl": "wonderful", "efficent": "efficient", "efficiant": "efficient", "efficeint": "efficient",
    "prompt": "prompt", "promt": "prompt", "quikc": "quick", "quik": "quick", "quickl": "quickly",
    "explaned": "explained", "explianed": "explained", "explaind": "explained", "explainded": "explained",
    "undestood": "understood", "understod": "understood", "understoood": "understood", "uderstood": "understood",
    "recomend": "recommend", "reccomend": "recommend", "recommed": "recommend", "reccommend": "recommend",
    "appriciate": "appreciate", "appreciat": "appreciate", "apreciate": "appreciate", "appreciatd": "appreciated",
    "waitd": "waited", "waitig": "waiting", "watied": "waited", "waitied": "waited", "wated": "waited",
    "rudee": "rude", "ruude": "rude", "ignord": "ignored", "ignroed": "ignored", "ignred": "ignored",
    "rediculous": "ridiculous", "ridiclous": "ridiculous", "ridicuous": "ridiculous",
    "unaceptable": "unacceptable", "unacceptble": "unacceptable", "unnaceptable": "unacceptable",
    "overcharge": "overcharge", "overchaged": "overcharged", "overcharg": "overcharge",
    "exspensive": "expensive", "expensiv": "expensive", "expesive": "expensive", "expnsive": "expensive",
    "chasd": "chased", "chasng": "chasing", "chaisng": "chasing", "chsing": "chasing",
    "evrything": "everything", "everyting": "everything", "everythng": "everything",
    "stafff": "staff", "stff": "staff", "staf": "staff", "sttaff": "staff",
    "servce": "service", "sevice": "service", "servcie": "service", "servise": "service", "seervice": "service",
    "comunication": "communication", "communcation": "communication", "communiation": "communication",
    "comminication": "communication", "commuication": "communication", "communicaton": "communication",
    "thnx": "thanks", "thankyou": "thank you", "thnks": "thanks", "thanx": "thanks", "thanls": "thanks",
}



# ---- additions found by reviewing mis-classified comments (domain-specific) ----
POS_WORDS.update({
    # customers here often set up a positive surprise: "was dreading it ... turned out cheap"
    "expecting": 0.6, "expected": 0.3, "braced": 0.8, "dreading": 0.8, "dreaded": 0.8, "feared": 0.8,
    "nervous": 0.5, "baffled": 0.3, "surprised": 0.5, "surprise": 0.2, "shocked": 0.5, "shock": 0.3,
    "fastest": 2, "quickest": 2, "simpler": 1.5, "faster": 1.2, "quicker": 1.2, "smoother": 1.5,
    "better": 1, "beauty": 1.5, "slick": 1.5, "bonus": 1.2, "win": 1.2, "respect": 1.2, "spot": 0.3,
    "sensible": 1, "lucid": 1.5, "pitched": 0.5, "rolled": 0.3, "hiccups": 0, "gladly": 1, "beaut": 1.5,
    "cleaner": 0.8, "cleaned": 0.8, "washed": 0.8, "pinged": 0.8, "texted": 0.8, "texts": 0.5, "sms": 0.3,
    "updates": 0.5, "update": 0.3, "keeping": 0.2, "dinkum": 0.3, "sweet": 1.2, "stoked": 2,
    "rocks": 1.5, "rocking": 1.5, "slayed": 1.5, "nailed": 1.8, "sorted": 1.2, "ripper": 2,
    "professionalism": 1.5, "courtesy": 0.8, "manners": 0.8, "politeness": 1.2, "civil": 0.8,
    "refreshingly": 1.5, "relief": 1.2, "peace": 0.8, "reassurance": 1.2, "confident": 1.2,
    "confidence": 1.2, "trust": 1.2, "trusted": 1.2, "honesty": 1.5, "integrity": 1.5,
    "transparency": 1.5, "clarity": 1.2, "plainly": 0.8, "straight": 0.3, "straightforwardly": 1.2,
    "approachable": 1.5, "likeable": 1.2, "likable": 1.2, "pleasing": 1.2, "cared": 1.0,
    "listened": 1.2, "listening": 0.8, "attention": 0.5, "interest": 0.3, "interested": 0.3,
    "patience": 1.5, "empathy": 1.5, "empathetic": 1.5, "understanding": 1.2, "sympathetic": 1.2,
})
NEG_WORDS.update({
    "hurdle": 1.0, "hurdles": 1.0, "wonder": 0.6, "wondering": 0.9, "guessing": 1.0, "guess": 0.3,
    "sarcastic": 1.5, "sarcasm": 1.5, "snail": 1.8, "glacier": 1.5, "glaciers": 1.5, "crazy": 1.2,
    "jammed": 1.5, "shakes": 1.3, "shudders": 1.5, "shuddering": 1.5, "vibration": 1.3, "vibrating": 1.3,
    "pulls": 0.8, "beeps": 1.2, "beeping": 1.2, "crackles": 1.3, "overheating": 1.5, "overheats": 1.5,
    "smear": 1.2, "smears": 1.2, "smearing": 1.2, "streaks": 1.2, "streaking": 1.2, "bent": 1.3,
    "stalls": 1.5, "stalled": 1.5, "squeaks": 1.2, "squeaking": 1.2, "squealing": 1.3, "grinding": 1.3,
    "clunk": 1.3, "clunking": 1.3, "sticks": 0.9, "stuck": 0.8, "vanished": 1.5, "unacknowledged": 1.5,
    "acknowledged": 0.0, "stuffed": 1.0, "juggling": 0.8, "tiring": 1.2, "exhausting": 1.5, "skint": 0.0,
    "dunno": 1.2, "shrug": 1.5, "shrugged": 1.5, "interruption": 1.5, "nuisance": 1.8, "bother": 0.5,
    "bothering": 0.8, "dispute": 1.5, "disputed": 1.5, "disputing": 1.5, "discrepancy": 1.5,
    "variance": 1.0, "differed": 1.2, "costly": 1.5, "surprises": 0.5, "extras": 0.8, "extra": 0.4,
    "slog": 1.2, "shabby": 1.5, "wobbly": 1.0, "beaten-up": 0.3, "crumbling": 1.2, "pothole": 1.2,
    "empty": 0.8, "timed": 0.3, "refreshing": 0.0, "kicking": 0.6, "kindly": 0.4, "please": 0.6,
    "improve": 1.2, "improvement": 0.8, "why": 0.8, "seriously": 0.5, "honestly": 0.2, "sorry": 0.5,
    "apologise": 0.3, "apologised": 0.3, "apology": 0.3, "refund": 0.5, "credit": 0.4, "invoice": 0.1,
    "hard": 0.3, "harder": 0.5, "difficult": 1.0, "difficulty": 1.0, "complicated": 1.0, "confusion": 1.2,
    "still": 0.7, "again": 0.4, "yet": 0.2, "meanwhile": 0.5, "apparently": 0.6, "supposedly": 1.2,
    "allegedly": 1.0, "reckon": 0.2, "lap": 0.3, "arrears": 0.5, "stranded": 1.5, "idle": 1.0,
    "unattended": 1.2, "unsupervised": 1.2, "tardy": 1.2, "belated": 1.2, "behind": 0.6,
    "slower": 1.3, "longer": 0.8, "quoted": 0.2, "charged": 0.5, "charge": 0.2, "charges": 0.3,
    "fee": 0.4, "fees": 0.4, "billed": 0.5, "levied": 0.8, "gauged": 1.2, "gouged": 2.0,
    "swindled": 2.0, "fleeced": 2.0, "conned": 2.0, "duped": 2.0, "tricked": 2.0, "sneaky": 1.8,
    "shifty": 1.8, "dodgy": 1.8, "suspicious": 1.2, "fishy": 1.5, "shonky": 2.0, "cowboys": 2.0,
    "amateur": 1.5, "amateurs": 1.5, "amateurish": 1.5, "clowns": 2.0, "circus": 1.8, "joke": 1.5,
    "laughable": 1.8, "farcical": 1.8, "pitiful": 1.8, "lousy": 1.8, "dud": 1.5, "lemon": 1.5,
    "junk": 1.8, "trash": 1.8, "scrap": 1.0, "wreck": 1.5, "wrecked": 1.8, "trashed": 2.0,
    "butchered": 2.0, "hacked": 1.2, "mangled": 2.0, "gouge": 2.0,
})
# Standard words that carry a negative reading in this domain only when reported as unresolved
PHRASES.update({
    "really?": -1.2, "seriously?": -1.2, "are you serious": -1.8, "you serious": -1.5, "good one": -0.3,
    "snail pace": -2, "snail's pace": -2, "move quicker": -1.8, "move faster": -1.8, "moved quicker": -1.8,
    "longer than": -1.5, "took double": -1.8, "took triple": -2, "took twice": -1.8, "double the time": -1.8,
    "double that": -1.5, "instead of": -0.6, "far longer": -1.8, "way longer": -1.8, "much longer": -1.5,
    "didn't before": -1.8, "wasn't like that before": -2, "weren't like that before": -2, "never did": -1.5,
    "before the service": -0.6, "after the service": -0.3, "since the service": -0.8, "since the work": -0.8,
    "hasn't": -0.8, "hadn't": -0.6, "haven't": -0.4, "can't be right": -1.5, "can't believe": -0.3,
    "hard to start": -1.8, "timed out": -1.8, "time out": -1.2, "kept refreshing": -1.5,
    "lost my appointment": -2, "lost my booking": -2, "lost the booking": -2, "booking was lost": -2,
    "twice lost": -2, "not in their system": -1.8, "not in the system": -1.8, "no booking": -1.8,
    "ring-around": -1.8, "ring around": -1.5, "act like you're a bother": -2, "like you're a bother": -2,
    "like a bother": -2, "customers are an interruption": -2, "an interruption": -1.5,
    "surprise extras": -1.8, "costly surprises": -2, "surprise fee": -1.8, "surprise charge": -1.8,
    "surprise charges": -1.8, "surprise costs": -1.8, "surprise bill": -1.8, "surprise on the bill": -1.8,
    "had extra": -1.2, "had an extra": -1.2, "extra fee": -1.5, "extra charge": -1.5, "extra charges": -1.5,
    "extra on the": -1.2, "extras on": -1.2, "nearly $": -0.3, "wish the": -0.8, "just wish": -1.2,
    "hard sell": -1.8, "didn't greet": -1.5, "nobody greeted": -1.8, "nobody thanked": -1.5, "wasn't any": -1.2,
    "wasn't a": -0.2, "a shrug": -1.5, "got a shrug": -1.8, "one word": -0.8, "answered with one word": -1.8,
    "ask someone else": -1.8, "not welcoming": -1.8, "not welcoming at all": -2, "not great": -1.5,
    "not a great": -1.5, "not a great first impression": -2, "first impression": 0.2,
    "no accessories": -1.8, "no spare key": -1.8, "no roof racks": -1.8, "no change table": -1.2,
    "was wet": -1.2, "lock was broken": -1.8, "was broken": -1.5, "is broken": -1.5, "still wearing": -1.5,
    "half-fitted": -1.8, "still being washed": -1.5, "had to be ordered": -1.5, "must be a very": -0.5,
    "fixed nothing": -2, "fixed? ha": -2, "ha.": -0.5, "ha,": -0.3, "welcome to the neighbourhood": -0.5,
    "great planning": -1.8, "great new system": -1.0, "top notch": 2,
    "thanks for the radio silence": -2, "cheers for nothing": -2, "thanks for nothing": -2,
    "cheers for letting me know nothing": -2, "any chance of": -0.8, "any chance": -0.5,
    "would be grateful if": -0.3, "would appreciate a": -0.3, "would appreciate": -0.3, "i would be grateful": -0.3,
    "kindly review": -1.2, "kindly check": -1.0, "kindly provide": -1.0, "please review": -1.0,
    "please check": -1.0, "please explain": -1.2, "please improve": -1.8, "please call": -0.8,
    "to whom it may concern": -1.0, "dear sir": -0.3, "dear sir/madam": -0.5, "formal complaint": -2,
    "i wish to complain": -2, "i wish to raise": -1.5, "wish to lodge": -1.8, "lodge a complaint": -2,
    "i am writing to": -0.5, "i'm writing to": -0.5, "i wish to thank": 2, "wish to thank": 2,
    "like to thank": 1.8, "want to thank": 1.8, "just wanted to thank": 1.8, "wanted to say thanks": 1.8,
    "wanted to thank": 1.8, "made my day": 2, "made our day": 2, "before i finished": 1.2,
    "before i'd finished": 1.2, "before i had finished": 1.2, "done before": 1.2, "back before": 1.2,
    "well below": 1.0, "well within": 1.2, "a lot less": 1.2, "lot less than": 1.5, "much less": 1.2,
    "a lot cheaper": 1.8, "wasn't expecting it": 1.5, "wasn't expecting it back": 1.5, "didn't expect": 1.0,
    "didn't have to ring": 1.8, "didn't have to call": 1.8, "didn't have to": 0.8, "not once did": 0.8,
    "never had to wonder": 1.8, "never had to": 1.2, "no need to": 1.0, "not need to call": 1.5,
    "i am not need": 1.2, "not need to": 1.0, "kept coming without": 1.8, "without me asking": 1.8,
    "without being asked": 1.8, "without having to": 1.5, "before i could": 1.2, "before i had to": 1.5,
    "great that they": 1.0, "good that they": 1.0, "glad they": 1.2, "glad to": 1.0, "so glad": 1.5,
    "up front": 1.0, "upfront": 1.2, "straight up": 0.5, "good enough for me": 1.0, "all i could ask": 1.5,
    "all i can ask": 1.5, "couldn't ask for": 2, "could not ask for": 2, "can't ask for": 2,
    "wouldn't hesitate": 2, "would not hesitate": 2, "wouldn't go anywhere else": 2, "won't go anywhere else": 2,
    "don't need to look elsewhere": 2, "never going elsewhere": 2, "not going elsewhere": 2,
    "stuck with": 0.5, "stuck by": 1.2, "stuck around": 0.5, "loyal": 1.2, "loyalty": 1.0,
    "didn't break the bank": 1.8, "didn't cost": 1.2, "cost less": 1.5, "cost me less": 1.5,
    "cheap as chips": 1.8, "walked out relieved": 1.8, "relieved": 1.5, "pretty relieved": 1.8,
    "not bad at all": 1.5, "not so keen": -1.2, "not keen": -1.2, "not a fan": -1.5,
    "i left with": 0.3, "left with change": 1.5, "change in my pocket": 1.5,
    "ahead of": 0.3, "well ahead": 1.2, "way ahead": 1.2, "back to me": 0.3, "back in under": 1.5,
    "keys back": 0.5, "keys in hand": 1.0, "ready that afternoon": 1.5, "ready that morning": 1.5,
    "same afternoon": 1.2, "same morning": 1.2, "same day": 1.0, "next day": 0.3, "same arvo": 1.2,
    "showed up the next day": 1.2, "lined up": 0.3, "line up": 0.3, "carry on": 0.3, "could line up": 0.8,
    "less time than": 1.5, "took less time": 1.5, "shorter than": 1.2, "short wait": 1.2, "wait was short": 1.5,
    "wait was minimal": 1.5, "minimal wait": 1.5, "barely waited": 1.5, "hardly waited": 1.5,
    "hardly any wait": 1.5, "didn't wait": 1.2, "didn't have to wait": 1.5, "no queue": 1.0,
    "not rushed": 1.5, "didn't feel rushed": 1.5, "felt rushed": -1.5, "talked down to": -2, "talked down": -2,
    "not talked down": 1.8, "not lost": 1.2, "felt lost": -1.2, "feel lost": -1.2, "feeling lost": -1.2,
    "left out of the loop": -1.8, "out of the loop": -1.8, "left guessing": -1.8, "keep guessing": -1.8,
    "kept guessing": -1.8, "not left guessing": 1.8, "wasn't left guessing": 1.8,
    "no hiccups": 1.5, "without any hiccups": 1.8, "without hiccups": 1.8, "without a hitch": 1.8,
    "no hitches": 1.5, "no hitch": 1.5, "hitch free": 1.5, "hitch-free": 1.5, "free of fuss": 1.8,
    "free from fuss": 1.8, "entirely free": 1.2, "clockwork": 1.5, "ran like clockwork": 2,
    "run like clockwork": 2, "like clockwork": 1.5, "rolled along": 1.0, "rolled in": 0.3,
})


POS_WORDS.update({"smoothest": 2, "easiest": 1.8, "simplest": 1.8, "ace": 1.5, "tick": 1.0, "respected": 1.2,
                  "fuss-free": 1.5, "faff-free": 1.5, "unheard": 0.3, "wholesome": 1.2, "sound": 0.8,
                  "easygoing": 1.5, "chilled": 1.0, "laid-back": 1.0, "personable": 1.5, "ta": 0.5})
NEG_WORDS.update({"faff": 0.8, "dead": 0.0})
PHRASES.update({"dead simple": 1.8, "dead easy": 1.8, "zero faff": 1.5, "no faff": 1.5, "unheard of": 1.5,
                "wondering what i'd been worried about": 1.5, "wondering what i'd been worreid about": 1.5,
                "fell into place": 1.5, "big tick": 1.5, "in no time": 1.2, "hardly hard": 1.2,
                "not a number": 1.5, "like a person": 1.5, "treated me like a person": 2, "free of fuss": 1.8,
                "not many places": 1.0, "never happens": 0.8, "didn't blow the budget": 1.8,
                "easily covered": 1.2, "didn't make me": 0.3, "rang me at the exact time": 1.8,
                "exact time": 1.0, "never thought i'd say": 1.0, "from go to whoa": 1.0,
                "without me needing": 1.2, "without needing": 1.0, "unheard": 0.3})


EXPECTED_THINGS.update({"rings", "rung", "calling", "ringing", "replies", "responses", "calls", "answers",
                        "texts", "messages", "phones", "contacts", "updates", "update", "updating"})
NEG_WORDS.update({"pity": 1.2, "regrettably": 1.5, "pains": 1.2, "remains": 0.8, "despite": 0.6, "limited": 0.8,
                  "worn": 1.0, "decline": 1.2, "declined": 1.2, "unobtainable": 1.5, "unavailable": 1.2,
                  "tight": 0.2, "nowhere": 0.5, "cramped": 0.8, "outdated": 1.0, "dated": 0.6})
PHRASES.update({"would have been nice": -1.3, "would be nice": -1.0, "would've been nice": -1.3, "would have appreciated": -1.3,
                "would've appreciated": -1.3, "it would be nice": -1.3, "shouldn't have to": -1.8, "should not have to": -1.8,
                "not one update": -1.8, "not one call": -1.8, "not one text": -1.8, "no one rings back": -1.8,
                "nobody rings back": -1.8, "rings back": 0.0, "pains me to write": -2, "so it pains me": -2,
                "bit of patience": -1.0, "a bit of patience would": -1.8, "barely looked": -1.8, "barely spoke": -1.8,
                "turned into": -1.0, "real quick": -0.3, "some soap": 0.0, "no soap": -1.2, "no kid-friendly": -1.0,
                "need work": -1.2, "needs work": -1.2, "needed work": -1.2, "needs improvement": -1.5, "need improvement": -1.5,
                "could be improved": -1.3, "should be improved": -1.3, "waiting area": 0.0, "pity the": -1.5,
                "looking forward": 0.3, "not stupid": -0.5, "laughed": -1.0, "laughed at": -2, "smirked": -1.8})
TYPO_FIXES.update({"cancleled": "cancelled", "cancled": "cancelled", "cancelld": "cancelled", "arund": "around",
                   "worreid": "worried", "straightorward": "straightforward", "roguhly": "roughly", "acocunt": "account",
                   "siknt": "skint", "bokoed": "booked", "stck": "stuck", "hiaces": "hiaces"})

# --------------------------------------------------------------------------
# 2. TEXT CLEANING / TOKENISING
# --------------------------------------------------------------------------
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
RE_PHONE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")
RE_REGO = re.compile(r"\(\s*rego[^)]*\)", re.I)
RE_REQUEST = re.compile(r"\b(call|ring|email|text|contact|phone|message)\s+me\s*(on|at|via)?\s*[,:]?\s*(\d|[\w.+-]+@)?", re.I)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*|\d+|[!?.,;:]")


def clean_text(text: str) -> str:
    t = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = RE_EMAIL.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_REGO.sub(" ", t)
    t = RE_PHONE.sub(" ", t)
    t = RE_REQUEST.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fix_typos(tokens):
    return [TYPO_FIXES.get(tok, tok) for tok in tokens]


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\s{2,}|\n+", text)
    return [p for p in parts if p and p.strip()]


# --------------------------------------------------------------------------
# 3. SCORING
# --------------------------------------------------------------------------
CONTRACTIONS = [
    (r"\bdid not\b", "didn't"), (r"\bdoes not\b", "doesn't"), (r"\bdo not\b", "don't"),
    (r"\bwas not\b", "wasn't"), (r"\bwere not\b", "weren't"), (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"), (r"\bhas not\b", "hasn't"), (r"\bhave not\b", "haven't"),
    (r"\bhad not\b", "hadn't"), (r"\bcould not\b", "couldn't"), (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"), (r"\bwill not\b", "won't"), (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"), (r"\bi am\b", "i'm"), (r"\bit is\b", "it's"),
]
_CONTR_RE = [(re.compile(a), b) for a, b in CONTRACTIONS]

_PHRASE_KEYS = sorted(PHRASES, key=lambda k: (-len(k.split()), -len(k)))
_PHRASE_RE = [(k, re.compile(r"(?<![a-z'])" + re.escape(k) + r"(?![a-z'])")) for k in _PHRASE_KEYS]
_SARC_RE = [re.compile(p, re.I) for p in SARCASM_CUES]
_NEUTRAL_RE = [re.compile(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z'])") for p in NEUTRAL_PHRASES]

# sarcasm that is clear even without other negative words in the text
SARCASM_ALWAYS = [re.compile(p) for p in [
    r"\b(thanks|cheers|thank you) for (nothing|the (radio silence|silence|surprise|mystery)|letting me know nothing)",
    r"\b(five|5) stars? for\b", r"\bfor nothing\b", r"\bletting me know nothing\b", r"\bsarcastic\b",
    r"^(great|lovely|brilliant|wonderful|fantastic|perfect|super|nice|oh great|oh brilliant),? "
    r"(so|another|a|the|it|that|they|my|now|just|what|new|how)\b",
    r"\bgreat planning\b", r"\bgood one\b", r"\blove how\b", r"\bif you enjoy\b", r"\bat its finest\b",
    r"\bimpressive stuff\b", r"\bcool,? cool\b", r"\blovely (surprise|touch)\b", r"\bcan't wait to see what you\b",
    r"\btruly gripping\b", r"\bwonderful job,? truly\b", r"\bfantastic use of my time\b",
]]

# phrases that already contain their own negation / are not flipped by a preceding negator
_NEGATOR_RE = re.compile(r"(?<![a-z'])(no|not|never|nothing|nobody|without|neither|nor|zero|hardly|barely|"
                         r"\w+n't|can't|won't|cannot)(?![a-z'])")
RE_MONEY_MORE = re.compile(r"\$\s?\d[\d,]*(\.\d+)?\s*(more|extra|over|higher)\b")
RE_WEEKS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+|several|few|couple of|a couple of|"
                      r"nearly|almost|over|about)\s+(weeks|months|fortnights)\b")


def _is_caps(sentence: str) -> bool:
    letters = [c for c in sentence if c.isalpha()]
    return len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.7


def normalise(sentence: str) -> str:
    s = sentence.lower()
    tokens = fix_typos(RE_TOKEN.findall(s))
    s = " ".join(tokens)
    s = re.sub(r"\s+([!?.,;:])", r"\1", s)
    for rx, rep in _CONTR_RE:
        s = rx.sub(rep, s)
    return s


def score_sentence(sentence: str):
    """Return (pos, neg, neutral_hits, hits) for one sentence."""
    shouting = _is_caps(sentence)
    s_fixed = normalise(sentence)

    pos = neg = 0.0
    hits = []

    # --- regex features
    if RE_MONEY_MORE.search(s_fixed):
        neg += 1.5; hits.append(("$ more than quoted", -1.5))
    if RE_WEEKS.search(s_fixed):
        neg += 1.0; hits.append(("weeks/months", -1.0))

    # --- phrases (consumed so single words are not counted twice); negation-aware
    consumed = s_fixed
    for key, rx in _PHRASE_RE:
        w = PHRASES[key]
        for m in list(rx.finditer(consumed)):
            before = consumed[:m.start()].split(",")[-1].split()[-4:]
            negated = (not _NEGATOR_RE.match(key.split()[0])) and any(_NEGATOR_RE.fullmatch(t) for t in before)
            ww = w
            if negated:
                ww = -0.8 * w if w > 0 else 0.7 * (-w)
            if shouting:
                ww *= 1.2
            if ww > 0:
                pos += ww
            else:
                neg += -ww
            hits.append((("NOT " if negated else "") + key, round(ww, 2)))
        consumed = rx.sub(lambda m: " " * len(m.group(0)), consumed)

    # --- neutral wording
    neutral_hits = sum(1 for rx in _NEUTRAL_RE if rx.search(consumed))

    # --- single words with negation / intensifier handling
    toks = RE_TOKEN.findall(consumed)
    negate_left = 0
    intensity = 1.0
    for tok in toks:
        if tok in {".", "!", "?", ";", ":", ","}:
            negate_left = 0
            intensity = 1.0
            continue
        bare = tok.replace("'", "")
        if tok in NEGATORS or bare in NEGATORS or tok.endswith("n't"):
            negate_left = NEGATION_WINDOW
            continue
        if tok in INTENSIFIERS:
            intensity = INTENSIFIERS[tok]
            continue
        if tok in DIMINISHERS:
            intensity = DIMINISHERS[tok]
            continue

        w = 0.0
        if negate_left and tok in EXPECTED_THINGS:
            neg += 1.3 * intensity                      # "no call", "never updated" -> service failure
            hits.append((f"NEG+{tok}", -1.3))
            negate_left = 0
            intensity = 1.0
            continue
        if tok in POS_WORDS:
            w = POS_WORDS[tok]
        elif tok in NEG_WORDS:
            w = -NEG_WORDS[tok]

        if w != 0:
            w *= intensity
            if shouting:
                w *= 1.2
            if negate_left:
                w = -0.8 * w if w > 0 else 0.7 * (-w)
                negate_left = 0
            if w > 0:
                pos += w
            else:
                neg += -w
            hits.append((tok, round(w, 2)))
            intensity = 1.0
        else:
            if tok in NEG_PASSTHROUGH:
                continue                                 # pass-through words keep the negation window open
            if negate_left:
                negate_left -= 1
            intensity = 1.0

    return pos, neg, neutral_hits, hits


def classify(comment, debug=False):
    """Return 'Negative' / 'Okay' / 'Positive' for one comment ('' if the comment is blank)."""
    if comment is None or (isinstance(comment, float) and pd.isna(comment)) or not str(comment).strip():
        return ""
    text = clean_text(comment)
    if not text:
        return ""

    sentences = split_sentences(text)
    scored = [score_sentence(s) for s in sentences]
    sent_norm = [normalise(s) for s in sentences]
    text_neg = sum(x[1] for x in scored)

    pos_tot = neg_tot = 0.0
    neutral_tot = 0
    for i, (p, n, nh, hits) in enumerate(scored):
        prev_n = scored[i - 1][1] if i > 0 else 0
        next_n = scored[i + 1][1] if i + 1 < len(scored) else 0
        neighbour_neg = n + prev_n + next_n
        cue = any(rx.search(sent_norm[i]) for rx in _SARC_RE)
        always = any(rx.search(sent_norm[i]) for rx in SARCASM_ALWAYS)
        stripped = re.sub(r"[^a-z' ]", "", sent_norm[i]).strip()
        words = stripped.split()
        lone_praise = bool(words) and len(words) <= 3 and words[0] in SARCASM_SINGLES and i > 0 \
            and (prev_n >= 1.0 or text_neg >= 1.5)
        if always or (cue and (n > 0 or neighbour_neg >= 1.0)) or (lone_praise and neighbour_neg >= 0.8):
            # irony: any praise in this sentence is criticism
            neg_tot += max(p, 0) + n + 1.5
            if debug:
                print("  sarcasm ->", sentences[i])
            continue
        pos_tot += p
        neg_tot += n
        neutral_tot += nh

    net = pos_tot - neg_tot
    big, small = max(pos_tot, neg_tot), min(pos_tot, neg_tot)
    if debug:
        print(f"  pos={pos_tot:.2f} neg={neg_tot:.2f} neutral={neutral_tot}")
        for sc in scored:
            print("   ", sc[3])

    if big < 1.0:
        return "Okay"                                   # little or no sentiment signal
    if small >= 1.0 and small / big >= 0.55:
        return "Okay"                                   # mixed: both sides present in similar amounts
    if neutral_tot and abs(net) < 2.0:
        return "Okay"                                   # lukewarm wording
    if net >= 1.0:
        return "Positive"
    if net <= -1.0:
        return "Negative"
    return "Okay"


def main(in_csv=IN_CSV, out_csv=OUT_CSV):
    # dtype=str + keep_default_na=False => every other column is written back exactly as it was read
    df = pd.read_csv(in_csv, keep_default_na=False, dtype=str)
    df[NEW_COL] = df["Comment"].apply(classify)
    df.to_csv(out_csv, index=False)
    print(df[NEW_COL].replace("", "(blank comment)").value_counts())


if __name__ == "__main__":
    main(*sys.argv[1:])
