"""
memory_engine.py
প্রজেক্ট জারা - লং-টার্ম মেমোরি ইঞ্জিন।
JSON ফাইল থেকে মেমোরি লোড/সেভ/সার্চ/ডিলিট করার সম্পূর্ণ লজিক এখানে থাকে,
সাথে করাপ্টেড বা মিসিং ফাইলের জন্য সেফটি এরর হ্যান্ডলিং।
"""

import json
import os
from typing import Any, Dict, List

from config import MEMORY_FILE


def load_memory() -> Dict[str, List[Dict[str, str]]]:
    """
    মেমোরি JSON ফাইল থেকে ডেটা লোড করে।
    ফাইল না থাকলে বা করাপ্ট/ইনভ্যালিড হলে প্রোগ্রাম ক্র্যাশ না করে খালি ডিকশনারি রিটার্ন করে।
    """
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                # ফাইলে ভ্যালিড JSON থাকলেও তা ডিকশনারি না হলে সেফলি রিসেট
                return {}
            return data
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        # করাপ্টেড ফাইল বা রিড এরর হলে প্রোগ্রাম বন্ধ না করে সতর্কতা দেখিয়ে খালি মেমোরি রিটার্ন
        print(f"[সিস্টেম সতর্কতা] মেমোরি ফাইল লোড করতে সমস্যা হয়েছে ({e}). একটি নতুন মেমোরি সেশন শুরু করা হচ্ছে।")
        return {}


def save_memory(memory_data: Dict[str, Any]) -> bool:
    """
    মেমোরি ডেটা JSON ফাইলে সেভ করে।
    সেভ সফল হলে True, কোনো এরর হলে False রিটার্ন করে (ক্র্যাশ করে না)।
    """
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=4)
        return True
    except OSError as e:
        print(f"[সিস্টেম সতর্কতা] মেমোরি ফাইল সেভ করতে সমস্যা হয়েছে ({e}).")
        return False


def add_memory_entry(memory_data: Dict[str, Any], date_str: str, formatted_time: str, info: str) -> None:
    """নির্দিষ্ট তারিখের অধীনে একটি নতুন মেমোরি এন্ট্রি (সময় ও তথ্য) যোগ করে।"""
    if date_str not in memory_data:
        memory_data[date_str] = []
    memory_data[date_str].append({"time": formatted_time, "info": info})


def search_memory(memory_data: Dict[str, Any], date_str: str, keyword: str) -> List[str]:
    """নির্দিষ্ট তারিখের মেমোরির মধ্যে কীওয়ার্ড অনুযায়ী তথ্য খুঁজে লিস্ট আকারে রিটার্ন করে।"""
    if date_str not in memory_data:
        return []
    return [
        item["info"]
        for item in memory_data[date_str]
        if keyword.lower() in item["info"].lower()
    ]


def delete_today_memory(memory_data: Dict[str, Any], date_str: str) -> bool:
    """
    নির্দিষ্ট তারিখের মেমোরি ডেটা ক্লিয়ার করে এবং ফাইলে সেভ করে।
    ইতিমধ্যে ডেটা থাকলে True, খালি থাকলে False রিটার্ন করে।
    """
    if date_str in memory_data and memory_data[date_str]:
        memory_data[date_str] = []
        save_memory(memory_data)
        return True
    return False


def has_entries_today(memory_data: Dict[str, Any], date_str: str) -> bool:
    """নির্দিষ্ট তারিখে কোনো মেমোরি এন্ট্রি আছে কিনা তা যাচাই করে।"""
    return date_str in memory_data and len(memory_data[date_str]) > 0
