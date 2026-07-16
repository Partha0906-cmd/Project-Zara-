"""
learning_engine.py
প্রজেক্ট জারা - Self-Learning Logger।

যখনই intent_engine কোনো ইউজার-ইনপুটের জন্য কোনো ইনটেন্ট মেলাতে পারে না, সেই ইনপুটটা
এখানে JSON ফাইলে লগ হয়ে থাকে (raw text + normalized text + কতবার এসেছে + শেষ কখন এসেছে)।

উদ্দেশ্য: সময়ের সাথে সাথে বোঝা যাবে ইউজার কী কী বলছেন যেগুলো জারা এখনো বোঝে না —
সেগুলো রিভিউ করে intent_engine.INTENT_KEYWORDS-এ নতুন কীওয়ার্ড/সমার্থক শব্দ হিসেবে
যোগ করলে সিস্টেম ধীরে ধীরে নিজে থেকেই "স্মার্ট" হতে থাকবে। এটা কোনো অটোমেটিক মডেল-ট্রেনিং
না — এটা একটা মানুষ-সহায়ক ফিডব্যাক লুপ, সম্পূর্ণ অফলাইন ও লাইটওয়েট।

ফাইল ফরম্যাট (JSON):
{
    "entries": {
        "<normalized_text>": {
            "raw_examples": ["আসল ইনপুট ১", "আসল ইনপুট ২"],
            "count": 3,
            "last_seen": "2026-07-13 21:40:00"
        },
        ...
    }
}
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

from config import UNMATCHED_LOG_FILE

_MAX_RAW_EXAMPLES_PER_ENTRY = 3  # প্রতিটা এন্ট্রির জন্য কতগুলো আসল উদাহরণ-বাক্য রাখা হবে


def load_unmatched_log() -> Dict[str, Any]:
    """
    Unmatched-log JSON ফাইল থেকে ডেটা লোড করে।
    ফাইল না থাকলে বা করাপ্ট/ইনভ্যালিড হলে প্রোগ্রাম ক্র্যাশ না করে খালি স্ট্রাকচার রিটার্ন করে।
    """
    if not os.path.exists(UNMATCHED_LOG_FILE):
        return {"entries": {}}

    try:
        with open(UNMATCHED_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict) or "entries" not in data:
                return {"entries": {}}
            return data
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"[সিস্টেম সতর্কতা] Unmatched-log ফাইল লোড করতে সমস্যা হয়েছে ({e}). নতুন লগ শুরু করা হচ্ছে।")
        return {"entries": {}}


def save_unmatched_log(log_data: Dict[str, Any]) -> bool:
    """Unmatched-log ডেটা JSON ফাইলে সেভ করে। ব্যর্থ হলে ক্র্যাশ না করে False রিটার্ন করে।"""
    try:
        with open(UNMATCHED_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        return True
    except OSError as e:
        print(f"[সিস্টেম সতর্কতা] Unmatched-log ফাইল সেভ করতে সমস্যা হয়েছে ({e}).")
        return False


def log_unmatched(raw_text: str, normalized_text: str) -> None:
    """
    কোনো ইনটেন্ট না মেলা ইনপুট লগ করে। একই normalized_text আগে থেকে থাকলে count বাড়িয়ে
    দেওয়া হয় এবং raw_examples-এ (ডুপ্লিকেট এড়িয়ে, সর্বোচ্চ _MAX_RAW_EXAMPLES_PER_ENTRY টা)
    নতুন উদাহরণ যোগ হয়।
    """
    if not normalized_text:
        return

    log_data = load_unmatched_log()
    entries = log_data["entries"]

    if normalized_text not in entries:
        entries[normalized_text] = {
            "raw_examples": [raw_text],
            "count": 1,
            "last_seen": datetime.now().isoformat(),
        }
    else:
        entry = entries[normalized_text]
        entry["count"] += 1
        entry["last_seen"] = datetime.now().isoformat()
        if raw_text not in entry["raw_examples"] and len(entry["raw_examples"]) < _MAX_RAW_EXAMPLES_PER_ENTRY:
            entry["raw_examples"].append(raw_text)

    save_unmatched_log(log_data)


def get_top_unmatched(limit: int = 10) -> list:
    """
    সবচেয়ে বেশিবার আসা unmatched ইনপুটগুলো রিটার্ন করে, প্রথমে count (descending) দিয়ে
    সাজানো, আর count সমান হলে সবচেয়ে সাম্প্রতিক (last_seen descending) আগে থাকবে —
    যাতে পুরনো এন্ট্রি (যেমন সেশনের প্রথম দিকের কোনো টাইপো) নতুন, বেশি প্রাসঙ্গিক unmatched
    ইনপুটকে (যেমন "শেখাও" লিস্টে) চাপা না দিয়ে ফেলে।
    প্রতিটা আইটেম: (normalized_text, count, raw_examples, last_seen)
    """
    log_data = load_unmatched_log()
    entries = log_data["entries"]
    sorted_items = sorted(
        entries.items(),
        key=lambda kv: (kv[1]["count"], kv[1]["last_seen"]),
        reverse=True,
    )
    return [
        (norm_text, info["count"], info["raw_examples"], info["last_seen"])
        for norm_text, info in sorted_items[:limit]
    ]


def clear_unmatched_log() -> bool:
    """পুরো unmatched log খালি করে দেয় (যেমন রিভিউ করে কীওয়ার্ড যোগ করার পর)।"""
    return save_unmatched_log({"entries": {}})


def remove_unmatched_entry(normalized_text: str) -> bool:
    """
    একটা নির্দিষ্ট এন্ট্রি লগ থেকে সরিয়ে দেয় (যেমন Active Self-Learning-এ "শেখানো" হয়ে গেলে,
    যাতে সেটা বারবার রিভিউ-লিস্টে না দেখায়)।
    """
    log_data = load_unmatched_log()
    if normalized_text in log_data["entries"]:
        del log_data["entries"][normalized_text]
        return save_unmatched_log(log_data)
    return False
