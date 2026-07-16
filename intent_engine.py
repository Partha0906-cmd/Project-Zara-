"""
intent_engine.py
প্রজেক্ট জারা - সম্পূর্ণ অফলাইন Intent Recognition ইঞ্জিন।

কোনো API/ইন্টারনেট লাগে না — শুধু Python-এর built-in লাইব্রেরি (re, unicodedata,
difflib, math, collections) ব্যবহার করা হয়েছে, তাই Pydroid3-তে কোনো এক্সট্রা
pip install ছাড়াই চলবে।

কীভাবে কাজ করে (দুই-লেয়ার সিস্টেম):
  ১. normalize_text() ইউজারের বাক্য থেকে অদৃশ্য ক্যারেক্টার, এক্সট্রা স্পেস, পাংচুয়েশন সরিয়ে
     একটা পরিষ্কার ফর্মে আনে।
  ২. [Layer 1 — Keyword Matching] প্রতিটা ইনটেন্টের জন্য INTENT_KEYWORDS-এ অনেকগুলো
     সমার্থক শব্দ/বাক্যাংশ রাখা আছে। এক্সাক্ট ফ্রেজ ম্যাচ পেলে বেশি পয়েন্ট, আর সব শব্দ
     fuzzy-ভাবে (ছোটখাটো বানান ভুল সহ) পাওয়া গেলে কম পয়েন্ট। সবচেয়ে বেশি স্কোর পাওয়া
     ইনটেন্ট রিটার্ন হয়। টাই হলে অগ্রাধিকার-তালিকা (INTENT_PRIORITY) অনুযায়ী সিদ্ধান্ত হয়।
  ৩. [Layer 2 — TF-IDF Semantic Fallback] Layer 1 কিছুই না মেলাতে পারলে (কোনো কীওয়ার্ড
     ওভারল্যাপ নেই), একটা lightweight TF-IDF + Cosine Similarity লেয়ার বাক্যের সামগ্রিক
     "শব্দ-প্রোফাইল" প্রতিটা ইনটেন্টের উদাহরণ-বাক্যগুলোর centroid-এর সাথে তুলনা করে।
     মিল যথেষ্ট শক্তিশালী (থ্রেশহোল্ডের উপরে) হলে সেই ইনটেন্ট গ্রহণ করা হয় — এভাবে
     সম্পূর্ণ নতুন প্যারাফ্রেজ/বাক্য-গঠনও কিছুটা বোঝা সম্ভব হয়, keyword লিস্টে না থাকলেও।
     এটা সম্পূর্ণ deterministic ও অফলাইন — কোনো external model/library লাগে না।
"""

import re
import math
import json
import os
import unicodedata
import difflib
from collections import Counter
from typing import Dict, List, Optional, Tuple

import learning_engine
import embedding_engine
from config import LEARNED_KEYWORDS_FILE


# ---------------------------------------------------------
# টেক্সট নরমালাইজেশন
# ---------------------------------------------------------
_ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff"]
_PUNCTUATION_PATTERN = re.compile(r"[।,!?.\"'…]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    ইউজারের বাক্যকে ম্যাচিং-এর জন্য প্রস্তুত করে:
    - Unicode NFC normalize (ফোন কীবোর্ড/ফন্ট ভেদে গঠনগত পার্থক্য দূর করতে)
    - Zero-width/অদৃশ্য ক্যারেক্টার সরানো
    - পাংচুয়েশন সরানো ও এক্সট্রা স্পেস একত্র করা
    - lowercase (ইংরেজি অংশের জন্য)
    - 'কী'/'কি' বানান-ভেদ একত্র করা (দুটোই একই অর্থ বহন করে, কিন্তু ভিন্নভাবে লেখা হয় —
      এক না করলে কীওয়ার্ড ও semantic ম্যাচিং দুটোতেই মিসম্যাচ হতে পারে)
    """
    normalized = unicodedata.normalize("NFC", text)
    for ch in _ZERO_WIDTH_CHARS:
        normalized = normalized.replace(ch, "")
    normalized = normalized.lower()
    normalized = normalized.replace("কী", "কি")
    normalized = _PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


# ---------------------------------------------------------
# প্রতিটা ইনটেন্টের জন্য কীওয়ার্ড/সমার্থক শব্দের তালিকা
# (বিদ্যমান সব পুরনো কীওয়ার্ড অক্ষুণ্ণ রাখা হয়েছে, শুধু আরও সমার্থক শব্দ যোগ করা হয়েছে)
# ---------------------------------------------------------
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "memory_delete": [
        "মেমোরি ডিলিট", "মুছে ফেলো", "ক্লিয়ার করো", "ডিলিট করো",
        "মুছে দাও", "মুছে দে", "রিমুভ করো", "মেমোরি রিসেট", "ডেটা মুছে দাও",
        "সব মুছে দাও", "মেমোরি খালি করো", "পুরনো তথ্য মুছে দাও", "মেমোরি ফাঁকা করে দাও",
        "delete memory", "clear memory", "remove memory", "delete all memory",
        "clear all data", "erase memory", "wipe memory",
        "memory delete koro", "shob mucher dao", "shob delete koro", "memory clear koro",
    ],
    "memory_search": [
        "সার্চ করো", "খুঁজে দেখো", "খুঁজে বের করো", "মেমোরি থেকে খোঁজো",
        "খুঁজে দাও", "সার্চ দাও", "খোঁজ নাও", "মেমোরিতে আছে কিনা দেখো",
        "search memory", "find in memory", "search for", "look up in memory", "check if it's saved",
        "memory theke khojo", "search koro",
    ],
    "night_summary": [
        "গুড নাইট", "আজকের রিভিউ", "আজকের সামারি", "শুভরাত্রি",
        "আজকের দিনের রিভিউ", "আজকের রিপোর্ট দাও", "সামারি দাও",
        "আজকের পারফরম্যান্স বলো", "নাইট রিভিউ দাও", "দিনটা কেমন গেল সামগ্রিকভাবে",
        "good night", "today's review", "today's summary", "day review",
        "wrap up the day", "end of day summary", "how was my day overall",
        "ajker review", "ajker summary", "shuvo ratri",
    ],
    "memory_save": [
        "মনে রাখো", "মনে রাখিস", "সেভ করো", "save memory",
        "মনে রেখো", "মনে রেখো তো", "নোট করো", "লিখে রাখো", "মনে রাখবে",
        "সেভ করে রাখো", "মনে রাখতে হবে", "নোট রাখো", "ডায়েরিতে লিখে রাখো",
        "গুরুত্বপূর্ণ কথা মনে রেখো", "এইটা নোট করে নাও",
        "remember this", "remember that", "remember", "please remember",
        "note this", "keep in mind", "save this", "jot this down", "make a note of this",
        "mone rakho", "mone rekho", "note koro", "likhe rakho", "eta likhe rakho",
    ],
    "memory_recall": [
        "কি বলেছি", "আজকের মেমোরি", "কি মনে রেখেছ", "আজকে কি কি কাজ",
        "কি বলেছিলাম", "কি কি বলেছিলাম", "আজকে কি মনে আছে", "কি মনে আছে",
        "আজকে কি বলেছি", "মেমোরি দেখাও", "কি কি সেভ করেছিলাম", "আমার নোটগুলো দেখাও",
        "কি কি লিখে রেখেছিলে",
        "what did i say", "what did i tell you", "today's memory", "what do you remember",
        "show me my notes", "what have i saved",
        "ki bolechilam", "ajker memory bolo", "ki mone rekhecho", "ki ki save korechilam",
    ],
    "time_query": [
        "টাইম", "কটা বাজে", "time",
        "কয়টা বাজে", "ঘড়িতে কটা বাজে", "ঘড়িতে কয়টা বাজে", "এখন কটা বাজে",
        "এখন কয়টা বাজে", "সময়টা বলো", "সময় কত", "সময়টা কত", "কী সময় হলো",
        "ঘড়িতে সময় কত বাজে এখন", "এখন কয়টা", "টাইমটা একটু চেক করো",
        "what time is it", "current time", "what's the time", "tell me the time",
        "what's the current time now",
        "koyta baje", "ekhon koyta baje", "somoy koto", "ekhon somoy koto holo",
    ],
    "day_query": [
        "কী বার আজ", "আজ কী বার", "আজকে কী বার", "আজ কি বার", "আজকে কি বার",
        "কী বার", "বার কী আজ", "কী দিন আজ",
        "what day is it", "which day today", "what's today's day",
        "ajke ki bar", "aj ki bar",
    ],
    "hotword": [
        "হে জারা", "ওকে জারা", "hey zara", "okay zara",
    ],
    "routine_detail": [
        "রুটিনের মধ্যে কি কি আছে", "রুটিনে কি কি আছে", "রুটিন দেখাও",
        "পুরো রুটিন বলো", "রুটিন বিস্তারিত", "আজকের শিডিউল দেখাও",
        "সম্পূর্ণ রুটিন", "রুটিনটা বলো তো", "রুটিনে কি কি", "শিডিউলে কি কি আছে",
        "show routine", "full routine", "show schedule", "what's in the routine",
        "routine dekhao", "puro routine bolo", "ajker schedule dekhao",
    ],
    "routine_check": [
        "রুটিন", "আজকের দিন", "আজকের রুটিন", "আজ কী করব",
        "today's routine", "routine today",
        "ajker routine ki", "routine ki",
    ],
    "workout_plan_request": [
        "ওয়ার্কআউট কি", "ব্যায়াম কি", "ওয়ার্কআউট রুটিন", "আজকের ওয়ার্কআউট",
        "কোন ব্যায়াম", "আজকে কি ব্যায়াম", "কি ওয়ার্কআউট", "ওয়ার্কআউট আছে",
        "কোন ওয়ার্কআউট", "আজকে ওয়ার্কআউট করা উচিত", "কী ওয়ার্কআউট",
        "today's workout", "what's the workout", "which workout",
        "ajker workout ki", "ki workout", "kon workout",
    ],
    "workout_casual": [
        "ওয়ার্কআউট হলো", "ওয়ার্কআউট হল", "ব্যায়াম হলো", "জিম করলাম",
        "ওয়ার্কআউট ফাটাফাটি", "ওয়ার্কআউট দারুণ", "ওয়ার্কআউট ভালো",
        "জিম জমলো না", "জিম শেষ করলাম", "ওয়ার্কআউট শেষ করলাম",
        "চেস্ট ডে ছিল", "মাসল পেইন", "বডি পেইন হচ্ছে",
        "finished gym", "workout was great", "gym done", "finished workout",
        "gym sesh korlam", "workout darun holo", "gym jomlo na",
    ],
    "tired": [
        "টায়ার্ড", "ভালো লাগছে না", "ক্লান্ত", "খারাপ লাগছে",
        "এনার্জি নাই", "শরীর ভালো না", "ক্লান্তি লাগছে",
        "feeling tired", "i'm tired", "exhausted", "not feeling well",
        "klanto lagche", "tired lagche",
    ],
    "motivate": [
        "মোটিভেট", "মোটিভেশন", "এনার্জি দাও", "মোটিভেট করো",
        "ইন্সপায়ার করো", "চাঙ্গা করে দাও",
        "motivate me", "need motivation", "inspire me",
        "motivate koro", "energy dao",
    ],
    "day_update": [
        "দিনটা খুব ভালো", "ভালো গেল", "কাজ করলাম", "দিনটা দারুণ গেল",
        "আজকে ভালো ছিল", "আজকে প্রোডাক্টিভ ছিল",
        "day was great", "had a great day", "productive day",
        "dinta bhalo gelo", "aj bhalo chilo",
    ],
    "casual_chat": [
        "কী করছ", "কেমন কাটল", "কেমন আছ", "কি খবর", "কেমন চলছে",
        "how are you", "what are you doing", "how's it going",
        "kemon acho", "ki korcho", "kemon achis",
    ],
    "slacking_confession": [
        "আলতু ফালতু কাজ করছি", "ফাঁকিবাজি করছি", "সময় নষ্ট করছি",
        "ফেসবুক দেখছি", "রিলস দেখছি", "ইউটিউব দেখছি", "গেম খেলছি",
        "মোবাইল টিপছি", "পড়া ফেলে রেখেছি", "স্ক্রল করছি",
        "পড়তে ইচ্ছা করছে না", "পড়ায় মন নেই", "পড়া বাদ দিয়ে",
        "অন্য কাজ করছি", "টাইমপাস করছি", "আড্ডা দিচ্ছি",
        "watching youtube", "playing games", "wasting time", "scrolling",
        "watching reels", "on facebook",
        "youtube dekhchi", "reels dekhchi", "game khelchi", "somoy nosto korchi",
        "scroll korchi",
    ],
    "help": [
        "হেল্প", "সাহায্য", "কী করতে পারো", "তুমি কী করতে পারো", "কী কী করতে পারো",
        "কমান্ড লিস্ট", "কী কী কমান্ড আছে", "কী কী বলতে পারি", "গাইড দেখাও",
        "help", "what can you do", "list commands", "show commands", "how to use",
    ],
    "learning_review": [
        "কি কি বুঝতে পারনি", "কি কি বুঝতে পারো নি", "কি বুঝতে পারো না", "কি কি বোঝনি",
        "লার্নিং লগ দেখাও", "লার্নিং রিভিউ", "আনম্যাচড লগ দেখাও", "কোন কমান্ড বোঝনি",
        "কি কি শিখতে হবে", "কি কি মিস করছ",
        "show learning log", "what don't you understand", "unmatched log",
        "learning review", "ki ki bujhte parni", "learning log dekhao",
    ],
    "repeat_last": [
        "ওটা আবার বলো", "সেটা আবার বলো", "এটা আবার বলো", "আগেরটা আবার বলো",
        "আবার বলো তো", "আরেকবার বলো", "আরেকবার বলো তো", "রিপিট করো",
        "কি বললে আবার বলো", "একটু আবার বলো তো", "শেষেরটা আবার বলো",
        "repeat that", "say that again", "repeat again", "abar bolo",
    ],
    "teach_mode": [
        "শেখাও", "নতুন কিছু শেখাও", "তোমাকে শেখাব", "লার্নিং মোড", "কীওয়ার্ড যোগ করো",
        "নতুন শব্দ শেখাও", "শিখিয়ে দিই", "তোমাকে একটু শেখাই",
        "teach me", "teach mode", "learn new command", "add keyword",
        "shekhao", "sekhao", "shikhao", "shikhiye dao",
    ],
}

# ---------------------------------------------------------
# Active Self-Learning — persisted "শেখানো" কীওয়ার্ড লোড করা
# ---------------------------------------------------------
# হার্ডকোডেড INTENT_KEYWORDS (উপরের কিউরেটেড লিস্ট) ছাড়াও, ইউজার যদি "শেখাও" কমান্ডের
# মাধ্যমে রানটাইমে নতুন phrase যোগ করেন, সেগুলো LEARNED_KEYWORDS_FILE-এ জমা থাকে।
# প্রোগ্রাম শুরু হওয়ার সময় এখানে সেগুলো লোড করে INTENT_KEYWORDS-এ merge করে দেওয়া হয়,
# যাতে TF-IDF semantic index (নিচে) সহ সবকিছু এই "শেখা" phrase-গুলোও বিবেচনা করে।
try:
    if os.path.exists(LEARNED_KEYWORDS_FILE):
        with open(LEARNED_KEYWORDS_FILE, "r", encoding="utf-8") as _f:
            _learned = json.load(_f)
        for _intent_name, _phrases in _learned.items():
            if _intent_name in INTENT_KEYWORDS:
                for _phrase in _phrases:
                    if _phrase not in INTENT_KEYWORDS[_intent_name]:
                        INTENT_KEYWORDS[_intent_name].append(_phrase)
except (json.JSONDecodeError, UnicodeDecodeError, OSError, KeyError):
    pass  # করাপ্ট/অনুপস্থিত learned-file থাকলেও প্রোগ্রাম স্বাভাবিকভাবে চলবে, শুধু কিউরেটেড লিস্ট দিয়েই

# টাই-ব্রেকের জন্য অগ্রাধিকার (উপরের দিকেরটা বেশি গুরুত্বপূর্ণ)
INTENT_PRIORITY: List[str] = [
    "memory_delete", "memory_search", "night_summary", "memory_save",
    "memory_recall", "time_query", "day_query",
    "help", "learning_review", "repeat_last", "teach_mode",
    "slacking_confession",
    "routine_detail", "routine_check",
    "workout_plan_request", "workout_casual", "tired", "motivate",
    "day_update", "hotword", "casual_chat",
]

_EXACT_MATCH_WEIGHT = 2.0
_FUZZY_MATCH_WEIGHT = 1.0
_FUZZY_WORD_THRESHOLD = 0.78  # কতটা কাছাকাছি হলে টাইপো হিসেবে গণ্য হবে


_MIN_PREFIX_LEN_FOR_SUFFIX_MATCH = 3  # এর চেয়ে ছোট শব্দে প্রিফিক্স-ম্যাচ ব্যবহার করা হয় না (মিথ্যা মিল এড়াতে)


def _words_match(phrase_word: str, text_word: str) -> bool:
    """
    দুটো শব্দ "একই জিনিস বোঝাচ্ছে" কিনা যাচাই করে দুইভাবে:
    ১) প্রিফিক্স ম্যাচ — বাংলা প্রত্যয়/বিভক্তি (টা, তে, এর, শেষ ইত্যাদি) যোগ হলেও
       (যেমন 'জিম' বনাম 'জিমটা') মূল শব্দ মিলে যায়।
    ২) ফাজি রেশিও — ছোটখাটো বানান ভুল/টাইপো সহ্য করার জন্য (যেমন 'ওয়ার্কআউট' বনাম 'ওয়ার্কাউট')।
    """
    if len(phrase_word) >= _MIN_PREFIX_LEN_FOR_SUFFIX_MATCH and (
        text_word.startswith(phrase_word) or phrase_word.startswith(text_word)
    ):
        return True
    return difflib.SequenceMatcher(None, phrase_word, text_word).ratio() >= _FUZZY_WORD_THRESHOLD


def _fuzzy_phrase_present(phrase: str, text_words: List[str]) -> bool:
    """phrase-এর প্রতিটা শব্দ text_words-এর মধ্যে fuzzy-ভাবে (টাইপো/প্রত্যয় সহ্য করে) আছে কিনা যাচাই করে।"""
    for phrase_word in phrase.split():
        if not any(_words_match(phrase_word, tw) for tw in text_words):
            return False
    return True


_NEGATION_WORDS = ("না", "নাই", "নেই")
_NEGATION_WINDOW_CHARS = 6  # ম্যাচড ফ্রেজের ঠিক পরে কতটুকু টেক্সটের মধ্যে negation শব্দ খোঁজা হবে
_NEGATION_SENTENCE_WORD_LIMIT = 8  # এর চেয়ে ছোট বাক্যে standalone negation টোকেন থাকলেও negated ধরা হবে


def _phrase_has_builtin_negation(phrase: str) -> bool:
    """phrase নিজেই negation-শব্দ নিয়ে গঠিত কিনা (যেমন 'ভালো লাগছে না' — এটা একটা সম্পূর্ণ idiom)।"""
    return any(neg in phrase for neg in _NEGATION_WORDS)


def _is_locally_negated(phrase: str, normalized_text: str, text_words: List[str]) -> bool:
    """
    এই ম্যাচটা negated (উল্টো অর্থ) কিনা যাচাই করে দুইভাবে:
      ১. এক্সাক্ট ম্যাচ হলে: phrase-এর ঠিক পরে (ছোট window-এর মধ্যে) আলাদা negation শব্দ আছে কিনা
         (যেমন "ক্লান্ত না" — "ক্লান্ত"-এর ঠিক পরেই "না")
      ২. ফাজি ম্যাচ হলে (position-based check সম্ভব না, তাই broader net): পুরো বাক্যে standalone
         negation-টোকেন ("না"/"নাই"/"নেই" আলাদা শব্দ হিসেবে) আছে কিনা, এবং বাক্যটা ছোট
         (_NEGATION_SENTENCE_WORD_LIMIT-এর মধ্যে) — ছোট বাক্যে floating "না" প্রায় সবসময়ই
         কাছের content-word-কেই negate করে।

    phrase নিজেই ইতিমধ্যে negation নিয়ে গঠিত হলে (built-in idiom) এই চেক সম্পূর্ণ স্কিপ হয়,
    যাতে "ভালো লাগছে না"-এর মতো বৈধ ম্যাচ ভুলবশত বাতিল না হয়ে যায়।
    """
    if _phrase_has_builtin_negation(phrase):
        return False

    # ১. Exact-match position-based window চেক
    idx = normalized_text.find(phrase)
    if idx != -1:
        after_text = normalized_text[idx + len(phrase): idx + len(phrase) + _NEGATION_WINDOW_CHARS]
        if any(neg in after_text for neg in _NEGATION_WORDS):
            return True

    # ২. Fuzzy-match-এর জন্য broader sentence-level চেক (standalone negation টোকেন)
    if len(text_words) <= _NEGATION_SENTENCE_WORD_LIMIT:
        if any(w in _NEGATION_WORDS for w in text_words):
            return True

    return False


def _score_intent(keywords: List[str], normalized_text: str, text_words: List[str]) -> float:
    """একটা ইনটেন্টের জন্য মোট স্কোর হিসাব করে (এক্সাক্ট + fuzzy ম্যাচের সমষ্টি)। negated ম্যাচ বাদ যায়।"""
    score = 0.0
    for phrase in keywords:
        if phrase in normalized_text:
            if _is_locally_negated(phrase, normalized_text, text_words):
                continue  # "ক্লান্ত না" এর মতো negated ম্যাচ থেকে কোনো পয়েন্ট যোগ হবে না
            score += _EXACT_MATCH_WEIGHT
        elif _fuzzy_phrase_present(phrase, text_words):
            if _is_locally_negated(phrase, normalized_text, text_words):
                continue
            score += _FUZZY_MATCH_WEIGHT
    return score


# ---------------------------------------------------------
# [Layer 2] TF-IDF ভিত্তিক Semantic Similarity Fallback
# ---------------------------------------------------------
# Keyword matching সব বাক্য-গঠন কভার করতে পারে না। এই লেয়ারটা INTENT_KEYWORDS-এর সব
# উদাহরণ-বাক্য থেকে একটা ছোট TF-IDF vocabulary বানায়, প্রতিটা ইনটেন্টের জন্য একটা
# "centroid" ভেক্টর তৈরি করে (তার সব উদাহরণ-বাক্যের গড়), এবং রানটাইমে ইউজারের বাক্যের
# ভেক্টরের সাথে cosine similarity মেপে সবচেয়ে কাছাকাছি ইনটেন্ট খুঁজে বের করে।
# সম্পূর্ণ pure-Python (math + collections.Counter) — কোনো numpy/sklearn লাগে না।

_SEMANTIC_MATCH_THRESHOLD = 0.25  # Corpus Expansion-এর পর পুনরায় টিউন করা (৩৩২টা phrase-এর ভিত্তিতে)

# এই ইনটেন্টগুলো semantic fallback candidate থেকে বাদ — এগুলো "মেটা/স্ট্রাকচারাল" কমান্ড
# (রিপিট করা, শেখানো, হেল্প দেখানো) যাদের phrase-list ছোট এবং সাধারণ filler-word-heavy
# ("একটু", "বলো", "তো" ইত্যাদি), ফলে semantic centroid অতিরিক্ত ব্যাপক হয়ে যায় এবং
# সম্পূর্ণ-অসম্পর্কিত বাক্যও ভুলভাবে টেনে নেয়। এগুলোর জন্য শুধু keyword layer (exact/fuzzy)-ই
# নির্ভরযোগ্য — ব্যবহারকারী ঠিক এই কমান্ডগুলো বললে সেটা এমনিতেই keyword ম্যাচ করবে।
_SEMANTIC_EXCLUDED_INTENTS = {"repeat_last", "teach_mode"}


def _vectorize(words: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """শব্দের লিস্ট থেকে TF-IDF sparse ভেক্টর (dict আকারে) বানায়। vocabulary-তে নেই এমন শব্দ উপেক্ষা হয়।"""
    term_freq = Counter(words)
    return {w: count * idf[w] for w, count in term_freq.items() if w in idf}


def _l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """ভেক্টরকে ইউনিট-লেংথে (L2 norm = 1) নরমালাইজ করে, যাতে বাক্যের দৈর্ঘ্য cosine similarity-কে প্রভাবিত না করে।"""
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return vec
    return {w: v / norm for w, v in vec.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """দুটো sparse ভেক্টরের মধ্যে cosine similarity (0 থেকে 1, কতটা কাছাকাছি) হিসাব করে।"""
    common_words = set(vec_a) & set(vec_b)
    if not common_words:
        return 0.0
    dot_product = sum(vec_a[w] * vec_b[w] for w in common_words)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _build_semantic_index() -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    প্রোগ্রাম শুরু হওয়ার সময় একবার চলে: INTENT_KEYWORDS-এর সব উদাহরণ-বাক্য থেকে
    IDF টেবিল ও প্রতিটা ইনটেন্টের centroid ভেক্টর বানিয়ে রাখে (রানটাইমে বারবার
    রিক্যালকুলেট করার দরকার হয় না)।
    """
    # ধাপ ১: প্রতিটা কীওয়ার্ড-ফ্রেজকে একটা "ডকুমেন্ট" ধরে IDF হিসাব করা
    phrase_word_sets: List[set] = []
    for phrases in INTENT_KEYWORDS.values():
        for phrase in phrases:
            words = normalize_text(phrase).split()
            if words:
                phrase_word_sets.append(set(words))

    total_docs = len(phrase_word_sets)
    doc_freq: Counter = Counter()
    for word_set in phrase_word_sets:
        for w in word_set:
            doc_freq[w] += 1

    idf: Dict[str, float] = {
        w: math.log(total_docs / (1 + freq)) + 1.0 for w, freq in doc_freq.items()
    }

    # ধাপ ২: প্রতিটা ইনটেন্টের সব ফ্রেজের ভেক্টর যোগ করে centroid বানানো
    intent_centroids: Dict[str, Dict[str, float]] = {}
    for intent_name, phrases in INTENT_KEYWORDS.items():
        summed_vec: Counter = Counter()
        for phrase in phrases:
            words = normalize_text(phrase).split()
            for w, weight in _vectorize(words, idf).items():
                summed_vec[w] += weight
        intent_centroids[intent_name] = _l2_normalize(dict(summed_vec))

    return idf, intent_centroids


# মডিউল লোড হওয়ার সময় একবারই তৈরি হয় (runtime cost নেই, প্রতিটা কলে রিবিল্ড হয় না)
_IDF, _INTENT_CENTROIDS = _build_semantic_index()


def _semantic_fallback_intent(text_words: List[str]) -> Tuple[Optional[str], float]:
    """
    Keyword matching সম্পূর্ণ ব্যর্থ হলে TF-IDF cosine similarity দিয়ে সবচেয়ে কাছাকাছি
    ইনটেন্ট ও তার সিমিলারিটি-স্কোর রিটার্ন করে। কিছু না মিললে (None, 0.0)।
    """
    query_vec = _l2_normalize(_vectorize(text_words, _IDF))
    if not query_vec:
        return None, 0.0

    best_intent: Optional[str] = None
    best_similarity = 0.0
    for intent_name in INTENT_PRIORITY:
        if intent_name in _SEMANTIC_EXCLUDED_INTENTS:
            continue
        centroid = _INTENT_CENTROIDS.get(intent_name)
        if not centroid:
            continue
        similarity = _cosine_similarity(query_vec, centroid)
        if similarity > best_similarity:
            best_similarity = similarity
            best_intent = intent_name

    return best_intent, best_similarity


# ---------------------------------------------------------
# [Layer 3 — Elite Upgrade] Word Embedding ভিত্তিক Semantic Fallback
# ---------------------------------------------------------
# TF-IDF (Layer 2) শুধু "কোন শব্দ মিলল" দেখে — সমার্থক শব্দ (যেমন "ক্লান্ত" আর "দুর্বল")
# আলাদা string হওয়ায় সেটা ধরতে পারে না। এই লেয়ার Facebook fastText থেকে prune করা
# Bengali word vectors (embedding_engine.py) ব্যবহার করে প্রকৃত অর্থগত মিল মাপে, তাই
# keyword লিস্টে না থাকা সমার্থক শব্দও কিছুটা ধরতে পারে। embeddings ফাইল না থাকলে এই
# লেয়ার নিজে থেকেই নিষ্ক্রিয় থাকে (খালি centroid dict), প্রোগ্রাম স্বাভাবিকভাবেই চলবে।

_EMBEDDING_MATCH_THRESHOLD = 0.50  # টেস্ট করে টিউন করা — এর উপরে গেলে ভুল guess প্রায় শূন্য, নিচে নামালে false-positive বাড়ে

# ব্যাকরণগত/ফিলার শব্দ — এগুলো প্রায় অর্থহীন-কনটেন্ট (ক্রিয়ার কাল-চিহ্নিতকারী, সর্বনাম,
# পরসর্গ ইত্যাদি) কিন্তু কোনো একটা নির্দিষ্ট ইনটেন্টের phrase-list-এ বারবার এসে IDF-কে
# বিভ্রান্ত করতে পারে (IDF সেগুলোকে ভুলভাবে "distinctive" ভেবে বেশি weight দিয়ে দেয়,
# কারণ IDF শুধু "সারা corpus-এ কতবার এসেছে" মাপে, "শব্দটা আদৌ topic বহন করে কিনা" মাপে না)।
# তাই embedding centroid/query বানানোর সময় এগুলো সম্পূর্ণ বাদ দেওয়া হচ্ছে, IDF যাই বলুক না কেন।
_EMBEDDING_STOPWORDS = {
    "করছি", "করি", "করো", "করলাম", "করব", "করে", "করা", "হচ্ছে", "হয়েছে", "হয়", "হবে",
    "আছে", "নেই", "থাকে", "দাও", "দিন", "দিয়ে", "নিয়ে",
    "আমি", "তুমি", "আপনি", "আমার", "তোমার", "আপনার", "সে", "তার",
    "একটু", "অনেক", "খুব", "খুবই", "একদম", "মোটামুটি",
    "আজকে", "আজ", "কাল", "এখন", "তখন",
    "কি", "কী", "যে", "তো", "এই", "ওই", "সেই", "এটা", "ওটা", "সেটা",
}


def _filter_stopwords(words: List[str]) -> List[str]:
    """embedding vectorization-এর আগে ব্যাকরণগত/ফিলার শব্দ বাদ দেয়।"""
    filtered = [w for w in words if w not in _EMBEDDING_STOPWORDS]
    return filtered if filtered else words  # সব শব্দই স্টপওয়ার্ড হলে (যেমন খুব ছোট বাক্য), মূল লিস্টই ব্যবহার হবে


def _build_embedding_centroids() -> Dict[str, "embedding_engine.array"]:
    """
    প্রতিটা ইনটেন্টের সব কীওয়ার্ড-ফ্রেজের শব্দ-ভেক্টর IDF-ওয়েটেড গড় করে centroid বানায়
    (embedding না থাকলে খালি dict)। IDF-ওয়েটিং জরুরি — নাহলে "আজকে"/"খুব"-এর মতো সাধারণ
    শব্দ, যেগুলো প্রায় সব phrase-এ বারবার আসে, distinctive content word-কে ছাপিয়ে যায়
    (_IDF টেবিল TF-IDF layer-এর জন্য আগেই তৈরি করা আছে, এখানে পুনরায় ব্যবহার করা হচ্ছে)।
    """
    if not embedding_engine.is_available():
        return {}

    centroids = {}
    for intent_name, phrases in INTENT_KEYWORDS.items():
        all_words: List[str] = []
        for phrase in phrases:
            all_words.extend(normalize_text(phrase).split())
        all_words = _filter_stopwords(all_words)
        centroid = embedding_engine.weighted_sentence_vector(all_words, _IDF)
        if centroid is not None:
            centroids[intent_name] = centroid
    return centroids


# মডিউল লোড হওয়ার সময় একবারই তৈরি হয় (embeddings ফাইল না থাকলে খালি dict থাকবে)
_EMBEDDING_CENTROIDS = _build_embedding_centroids()


def _embedding_fallback_intent(text_words: List[str]) -> Tuple[Optional[str], float]:
    """
    Layer 1 (keyword) ও Layer 2 (TF-IDF) দুটোই ব্যর্থ হলে word-embedding cosine similarity
    দিয়ে সবচেয়ে কাছাকাছি ইনটেন্ট খোঁজে। embeddings ফাইল না থাকলে সবসময় (None, 0.0)।
    """
    if not _EMBEDDING_CENTROIDS:
        return None, 0.0

    filtered_words = _filter_stopwords(text_words)
    query_vec = embedding_engine.weighted_sentence_vector(filtered_words, _IDF)
    if query_vec is None:
        return None, 0.0

    best_intent: Optional[str] = None
    best_similarity = 0.0
    for intent_name in INTENT_PRIORITY:
        if intent_name in _SEMANTIC_EXCLUDED_INTENTS:
            continue
        centroid = _EMBEDDING_CENTROIDS.get(intent_name)
        if centroid is None:
            continue
        similarity = embedding_engine._cosine_similarity(query_vec, centroid)
        if similarity > best_similarity:
            best_similarity = similarity
            best_intent = intent_name

    return best_intent, best_similarity


def _passes_negation_gate(intent_name: str, normalized_text: str, text_words: List[str]) -> bool:
    """
    চূড়ান্ত সেফটি-গেট — keyword বা semantic, যে লেয়ার থেকেই ইনটেন্ট আসুক না কেন এটা প্রযোজ্য।
    বাক্যে standalone negation টোকেন (না/নাই/নেই) থাকলে (এবং ছোট বাক্য হলে), এই ইনটেন্ট তখনই
    গ্রহণযোগ্য যদি এর কোনো keyword ইতিমধ্যে built-in negation নিয়ে গঠিত হয়ে সত্যিই টেক্সটে
    ম্যাচ করে (যেমন "ভালো লাগছে না")। নাহলে suppress করা হয় — কারণ floating negation শব্দ
    প্রায় সবসময়ই কাছের content-word-কে negate করে, আর TF-IDF layer এটা নিজে থেকে বোঝে না।
    """
    if len(text_words) > _NEGATION_SENTENCE_WORD_LIMIT:
        return True
    if not any(w in _NEGATION_WORDS for w in text_words):
        return True

    for phrase in INTENT_KEYWORDS[intent_name]:
        if _phrase_has_builtin_negation(phrase) and phrase in normalized_text:
            return True
    return False


def detect_intent(user_input: str, log_if_unmatched: bool = True) -> Optional[str]:
    """
    ইউজারের বাক্য বিশ্লেষণ করে সবচেয়ে সম্ভাব্য ইনটেন্টের নাম রিটার্ন করে।
    কোনো ইনটেন্ট না মিললে None রিটার্ন করে (তখন ডিফল্ট/ফলব্যাক রেসপন্স ব্যবহার হবে)।

    তিন লেয়ারে কাজ করে:
      Layer 1: exact/fuzzy কীওয়ার্ড ম্যাচিং (দ্রুত, সবচেয়ে নির্ভরযোগ্য)
      Layer 2: Layer 1 ব্যর্থ হলে TF-IDF cosine-similarity semantic fallback
      Layer 3: Layer 2-ও ব্যর্থ হলে word-embedding cosine-similarity fallback (সমার্থক শব্দ ধরার জন্য)

    [Self-Learning] তিন লেয়ারের কোনোটাতেই কিছু না মিললে (এবং log_if_unmatched=True
    থাকলে) ইনপুটটা learning_engine-এর মাধ্যমে লগ হয়, যাতে পরে রিভিউ করে নতুন কীওয়ার্ড
    যোগ করা যায়। log_if_unmatched=False ব্যবহার হয় যেখানে একই ইনপুট একাধিকবার
    detect_intent() দিয়ে পাস করানো হয় (যেমন main.py-এর post-reply চেক), যাতে একই
    ইনপুটের জন্য ডবল-লগ না হয়।
    """
    normalized_text = normalize_text(user_input)
    if not normalized_text:
        return None

    text_words = normalized_text.split()

    best_intent: Optional[str] = None
    best_score = 0.0

    for intent_name in INTENT_PRIORITY:
        keywords = INTENT_KEYWORDS[intent_name]
        score = _score_intent(keywords, normalized_text, text_words)
        if score > best_score:
            best_score = score
            best_intent = intent_name
        # score == best_score হলে INTENT_PRIORITY-তে আগে থাকা ইনটেন্টই থেকে যায় (tie-break)

    # Layer 1 ব্যর্থ হলে (কোনো কীওয়ার্ড ওভারল্যাপ পাওয়া যায়নি) Layer 2 (TF-IDF semantic) ট্রাই করা হয়
    if best_intent is None:
        semantic_intent, semantic_similarity = _semantic_fallback_intent(text_words)
        if semantic_intent is not None and semantic_similarity >= _SEMANTIC_MATCH_THRESHOLD:
            best_intent = semantic_intent

    # Layer 2-ও ব্যর্থ হলে Layer 3 (word-embedding semantic) ট্রাই করা হয় — সমার্থক শব্দ
    # (keyword লিস্টে হুবহু নেই এমন) থাকলে এটা ধরতে পারে। embeddings ফাইল না থাকলে এই
    # কল সবসময় (None, 0.0) রিটার্ন করবে, তাই নিরাপদে চলবে।
    if best_intent is None:
        embedding_intent, embedding_similarity = _embedding_fallback_intent(text_words)
        if embedding_intent is not None and embedding_similarity >= _EMBEDDING_MATCH_THRESHOLD:
            best_intent = embedding_intent

    # [Negation Safety-Gate] যে লেয়ার থেকেই আসুক, floating negation-এর কারণে ভুল ইনটেন্ট
    # গ্রহণযোগ্য হলে এখানে বাতিল হয়ে যাবে
    if best_intent is not None and not _passes_negation_gate(best_intent, normalized_text, text_words):
        best_intent = None

    if best_intent is None and log_if_unmatched:
        learning_engine.log_unmatched(user_input, normalized_text)

    return best_intent


# ---------------------------------------------------------
# Active Self-Learning — রানটাইমে নতুন কীওয়ার্ড যোগ করা ("শেখাও" ফ্লো থেকে ব্যবহৃত হয়)
# ---------------------------------------------------------
def add_learned_keyword(intent_name: str, phrase: str) -> bool:
    """
    ইউজার-অনুমোদিত একটা নতুন phrase কোনো নির্দিষ্ট ইনটেন্টে যোগ করে:
      ১. ইন-মেমরি INTENT_KEYWORDS আপডেট হয় (এই সেশনেই সাথে সাথে কাজ করবে)
      ২. config.LEARNED_KEYWORDS_FILE-এ persist হয় (পরের বার প্রোগ্রাম চালু হলেও থাকবে)
      ৩. TF-IDF semantic index (_IDF, _INTENT_CENTROIDS) রিবিল্ড হয়, যাতে নতুন phrase
         semantic matching-এও অংশ নেয়

    intent_name ভুল/অস্তিত্বহীন হলে বা phrase আগে থেকেই থাকলে False রিটার্ন করে।
    """
    if intent_name not in INTENT_KEYWORDS:
        return False

    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase or normalized_phrase in INTENT_KEYWORDS[intent_name]:
        return False

    # ১. ইন-মেমরি আপডেট
    INTENT_KEYWORDS[intent_name].append(normalized_phrase)

    # ২. ডিস্কে persist করা (লোড-মডিফাই-সেভ প্যাটার্ন, safe error handling সহ)
    try:
        if os.path.exists(LEARNED_KEYWORDS_FILE):
            with open(LEARNED_KEYWORDS_FILE, "r", encoding="utf-8") as f:
                learned = json.load(f)
        else:
            learned = {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        learned = {}

    learned.setdefault(intent_name, [])
    if normalized_phrase not in learned[intent_name]:
        learned[intent_name].append(normalized_phrase)

    try:
        with open(LEARNED_KEYWORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(learned, f, ensure_ascii=False, indent=4)
    except OSError as e:
        print(f"[সিস্টেম সতর্কতা] শেখা কীওয়ার্ড ডিস্কে সেভ করতে সমস্যা হয়েছে ({e}), তবে এই সেশনে কাজ করবে।")

    # ৩. Semantic index (TF-IDF + embedding দুটোই) রিবিল্ড (global রিঅ্যাসাইনমেন্ট)
    global _IDF, _INTENT_CENTROIDS, _EMBEDDING_CENTROIDS
    _IDF, _INTENT_CENTROIDS = _build_semantic_index()
    _EMBEDDING_CENTROIDS = _build_embedding_centroids()

    return True
