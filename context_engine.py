"""
context_engine.py
প্রজেক্ট জারা - Short-term Conversation Context (RAM-ভিত্তিক, সেশন-লোকাল)।

zara_memory.json-এর "permanent" মেমোরি থেকে এটা সম্পূর্ণ আলাদা — এটা শুধু চলতি সেশনে
শেষ কয়েকটা টার্ন (ইউজারের কথা + ইনটেন্ট + জারার রিপ্লাই) মনে রাখে। প্রোগ্রাম বন্ধ হলে
বা সিস্টেম রিস্টার্ট হলে এই কনটেক্সট হারিয়ে যায় — এটাই উদ্দেশ্য, কারণ এটা "সাম্প্রতিক
কথোপকথনের সুতো" ধরে রাখার জন্য, স্থায়ী তথ্য সংরক্ষণের জন্য না।

এই মডিউল দুইটা কাজে ব্যবহার হয়:
  ১. Pronoun/Reference resolution — "এটা মনে রাখো", "ওটা আবার বলো" ইত্যাদিতে
     "এটা/ওটা/সেটা" ঠিক কোন আগের কথাকে নির্দেশ করছে সেটা বোঝা।
  ২. Follow-up প্রশ্ন সনাক্তকরণ — "কতক্ষণ?", "কেন?" এর মতো ছোট, একা দাঁড়ালে অস্পষ্ট
     বাক্য আগের টার্নের টপিকের সাথে যুক্ত করে বোঝা।
"""

from collections import deque
from typing import List, NamedTuple, Optional


class Turn(NamedTuple):
    """একটা একক কথোপকথন-টার্নের স্ন্যাপশট।"""
    raw_input: str
    normalized_input: str
    intent: Optional[str]
    reply: str


_MAX_HISTORY = 3
_history: "deque[Turn]" = deque(maxlen=_MAX_HISTORY)

# রেফারেন্স-শব্দ — এগুলো থাকলে বোঝা যায় ইউজার আগের কোনো কথাকে নির্দেশ করছেন
REFERENCE_WORDS = {
    "এটা", "এইটা", "ওটা", "সেটা", "ঐটা", "oita", "eta", "sheta", "eita", "seta",
}

# ফলো-আপ মার্কার — এই শব্দগুলো একা/ছোট বাক্যে এলে বোঝা যায় আগের টপিক নিয়েই প্রশ্ন
FOLLOWUP_MARKERS = {
    "কেন", "কতক্ষণ", "কীভাবে", "কিভাবে", "কবে", "তাহলে", "আচ্ছা",
    "আরেকটু", "কই", "মানে", "কোনটা", "কোথায়", "কেমনে",
}


def add_turn(raw_input: str, normalized_input: str, intent: Optional[str], reply: str) -> None:
    """একটা নতুন টার্ন হিস্টোরিতে যোগ করে। সবচেয়ে পুরনোটা _MAX_HISTORY পার হলে অটো বাদ পড়ে যায়।"""
    _history.append(Turn(raw_input, normalized_input, intent, reply))


def get_last_turn() -> Optional[Turn]:
    """
    বর্তমান টার্নের ঠিক আগের টার্ন রিটার্ন করে (সবচেয়ে সাম্প্রতিক অ্যাড-করা টার্ন)।
    হিস্টোরি খালি থাকলে None রিটার্ন করে (যেমন সেশনের প্রথম মেসেজেই)।
    """
    return _history[-1] if _history else None


def get_history() -> List[Turn]:
    """পুরো হিস্টোরি পুরনো থেকে নতুন ক্রমে লিস্ট আকারে রিটার্ন করে।"""
    return list(_history)


def clear() -> None:
    """হিস্টোরি সম্পূর্ণ খালি করে দেয় (যেমন স্লিপ মোডে গেলে — নতুন সেশনে পুরনো টপিক টেনে আনা উচিত না)।"""
    _history.clear()


def contains_reference_word(words: List[str]) -> bool:
    """ইনপুটে এটা/ওটা/সেটা জাতীয় রেফারেন্স-শব্দ আছে কিনা যাচাই করে।"""
    return any(w in REFERENCE_WORDS for w in words)


def is_short_followup(normalized_text: str, word_count_threshold: int = 4) -> bool:
    """
    ইনপুটটা ছোট (word_count_threshold-এর মধ্যে) এবং তাতে একটা ফলো-আপ মার্কার
    (কেন/কতক্ষণ/তাহলে ইত্যাদি) আছে কিনা যাচাই করে। এই ধরনের ইনপুট একা দাঁড়ালে অস্পষ্ট —
    আগের টার্নের প্রসঙ্গ ছাড়া বোঝা সম্ভব না।
    """
    words = normalized_text.split()
    if not words or len(words) > word_count_threshold:
        return False
    return any(w in FOLLOWUP_MARKERS for w in words)
