"""
config.py
প্রজেক্ট জারা - কনফিগারেশন ও কনস্ট্যান্ট মডিউল।
এখানে সমস্ত স্ট্যাটিক ধ্রুবক (ফাইল পাথ, সময়সীমা, রুটিন প্ল্যান) সংজ্ঞায়িত করা আছে।
"""

from typing import Dict, List

# মেমোরি ফাইলের পাথ
MEMORY_FILE: str = "zara_memory.json"

# Self-Learning Logger — যে ইনপুটগুলোর কোনো ইনটেন্ট মেলেনি, সেগুলো এখানে লগ হয়
UNMATCHED_LOG_FILE: str = "unmatched_log.json"

# Active Self-Learning — "শেখাও" কমান্ডের মাধ্যমে ইউজার-অনুমোদিত নতুন কীওয়ার্ড এখানে জমা হয়
# (হার্ডকোডেড INTENT_KEYWORDS থেকে আলাদা রাখা হয়েছে, যাতে কিউরেটেড লিস্ট অক্ষত থাকে এবং
# সমস্যা হলে শুধু এই ফাইলটা ডিলিট করেই "শেখা" কীওয়ার্ডগুলো রিসেট করা যায়)
LEARNED_KEYWORDS_FILE: str = "learned_keywords.json"

# Word Embeddings — Facebook fastText (cc.bn.300) থেকে prune করা Bengali word vectors।
# শুধু আমাদের vocab + সবচেয়ে সাধারণ ~১৫,০০০ শব্দ, 100-dim (মূল 300-dim থেকে truncate করা),
# ফাইল ফরম্যাট: প্রতি লাইনে "শব্দ<TAB>v1,v2,...,v100"। ফাইল না পাওয়া গেলে embedding layer
# নিষ্ক্রিয় থাকবে, প্রোগ্রাম ক্র্যাশ করবে না — শুধু TF-IDF fallback দিয়েই চলবে।
BENGALI_EMBEDDINGS_FILE: str = "bn_vectors_pruned.txt"

# --- সময়সীমা (মিনিট, মধ্যরাত থেকে গণনা করা) ---
SCHOOL_START: int = 10 * 60 + 40   # 10:40 AM
SCHOOL_END: int = 16 * 60 + 30     # 04:30 PM (সোম-বৃহস্পতি)
SATURDAY_SCHOOL_END: int = 14 * 60  # 02:00 PM (শনিবার শর্ট ডে)
REFRESH_END: int = 17 * 60 + 0     # 05:00 PM
TUITION_END: int = 19 * 60 + 0     # 07:00 PM

# টিউশন ডে (0=Monday ... 6=Sunday, Python weekday() অনুযায়ী)
# মঙ্গলবার(1), বৃহস্পতিবার(3), শনিবার(5), রবিবার(6)
TUITION_DAYS: List[int] = [1, 3, 5, 6]

# সাপ্তাহিক ওয়ার্কআউট রুটিন প্ল্যান
WORKOUT_PLAN: Dict[int, str] = {
    0: "Chest & Triceps (Focus: Maximizing push strength & direct triceps volume)",
    1: "Back & Biceps (Focus: Optimal pulling mechanics & deep muscle contraction across posterior chain)",
    2: "Legs & Abs (Focus: Heavy squats foundation for raw lower-body power & core structural integrity)",
    3: "Arms Hypertrophy (Focus: Isolated Biceps & Triceps stimulation to accelerate growth)",
    4: "Shoulder & Chest (Focus: Maximizing upper body width & overhead pressing power)",
    5: "Back & Shoulder (Focus: High-intensity workout for sharp V-taper aesthetic & bulletproof rear delts)",
    6: "Rest & Recovery (Focus: Mandatory rest for muscle protein synthesis, tissue repair & creatine saturation)",
}

# Python weekday() ইনডেক্স অনুযায়ী দিনের নাম (0=Monday ... 6=Sunday)
PYTHON_DAYS: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# একই ইনডেক্স অনুযায়ী বাংলা দিনের নাম (ইউজার-ফেসিং রেসপন্সে ব্যবহারের জন্য)
BENGALI_DAY_NAMES: List[str] = [
    "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার"
]

# --- Sleep Mode (24/7 hardware environment) কীওয়ার্ড ---
# এই শব্দগুলো বললে প্রোগ্রাম বন্ধ হবে না, শুধু sleep_mode = True হবে (এক্সাক্ট ম্যাচ)
SLEEP_TRIGGER_KEYWORDS: List[str] = ["exit", "quit", "বাই", "স্টপ"]

# স্লিপ মোডে থাকা অবস্থায় এই যেকোনো একটি হট-ওয়ার্ড থাকলে জারা জেগে উঠবে (partial/substring ম্যাচ)
WAKE_KEYWORDS: List[str] = ["হে জারা", "ওকে জারা", "জেগে ওঠো", "hey zara", "okay zara", "wake up", "jege otho"]

# --- ডে-টাইপ লেবেল (আগে "টিউশন ডে"/"নো-টিউশন ডে" ছিল) ---
LEARNING_DAY_LABEL: str = "লার্নিং ডে"       # মঙ্গল, বৃহস্পতি, শনি, রবি (TUITION_DAYS)
PRODUCTIVE_DAY_LABEL: str = "প্রোডাক্টিভ ডে"  # সোম, বুধ, শুক্র

# --- দৈনিক শিডিউল ব্লক (সময়, কাজ) — প্রতিটা এন্ট্রি একটা (time_range, label) টাপল ---
# সাধারণ লার্নিং ডে (মঙ্গল, বৃহস্পতি) — শনি/রবি আলাদা ওভাররাইড নিচে
SCHEDULE_LEARNING_DAY: List[tuple] = [
    ("5:00 AM - 7:00 AM", "ওয়ার্কআউট"),
    ("7:00 AM - 9:30 AM", "সেলফ-স্টাডি ও প্রোডাক্টিভ টাইম"),
    ("9:30 AM - 10:40 AM", "স্কুলের জন্য রেডি হওয়া"),
    ("10:40 AM - 5:00 PM", "স্কুল পিরিয়ড"),
    ("5:00 PM - 7:00 PM", "টিউশন পিরিয়ড"),
    ("7:00 PM - 9:00 PM", "স্টাডি"),
    ("9:00 PM - 10:30 PM", "নিজের এন্টারটেইনমেন্ট টাইম"),
]

# সাধারণ প্রোডাক্টিভ ডে (সোম, বুধ, শুক্র) — একই শিডিউল, শুধু ৫-৭টার স্লট ফ্রি টাইম
SCHEDULE_PRODUCTIVE_DAY: List[tuple] = [
    ("5:00 AM - 7:00 AM", "ওয়ার্কআউট"),
    ("7:00 AM - 9:30 AM", "সেলফ-স্টাডি ও প্রোডাক্টিভ টাইম"),
    ("9:30 AM - 10:40 AM", "স্কুলের জন্য রেডি হওয়া"),
    ("10:40 AM - 5:00 PM", "স্কুল পিরিয়ড"),
    ("5:00 PM - 7:00 PM", "ফ্রি টাইম"),
    ("7:00 PM - 9:00 PM", "স্টাডি"),
    ("9:00 PM - 10:30 PM", "নিজের এন্টারটেইনমেন্ট টাইম"),
]

# শনিবার ব্যতিক্রম (লার্নিং ডে, কিন্তু স্কুল শর্ট ডে ১০:৪০-২:০০)
SCHEDULE_SATURDAY: List[tuple] = [
    ("5:00 AM - 7:00 AM", "ওয়ার্কআউট"),
    ("7:00 AM - 9:30 AM", "সেলফ-স্টাডি ও প্রোডাক্টিভ টাইম"),
    ("9:30 AM - 10:40 AM", "স্কুলের জন্য রেডি হওয়া"),
    ("10:40 AM - 2:00 PM", "স্কুল পিরিয়ড (শনিবার শর্ট ডে)"),
    ("2:00 PM - 5:00 PM", "রেস্ট / মাই টাইম"),
    ("5:00 PM - 7:00 PM", "টিউশন পিরিয়ড"),
    ("7:00 PM - 9:00 PM", "স্টাডি"),
    ("9:00 PM - 10:30 PM", "নিজের এন্টারটেইনমেন্ট টাইম"),
]

# রবিবার ব্যতিক্রম (লার্নিং ডে, স্কুল বন্ধ থাকায় সকালের পর পুরোটাই মাই টাইম)
SCHEDULE_SUNDAY: List[tuple] = [
    ("5:00 AM - 7:00 AM", "ওয়ার্কআউট"),
    ("7:00 AM - 9:30 AM", "সেলফ-স্টাডি ও প্রোডাক্টিভ টাইম"),
    ("9:30 AM - 5:00 PM", "মাই টাইম (স্কুল বন্ধ)"),
    ("5:00 PM - 7:00 PM", "টিউশন পিরিয়ড"),
    ("7:00 PM - 9:00 PM", "স্টাডি"),
    ("9:00 PM - 10:30 PM", "নিজের এন্টারটেইনমেন্ট টাইম"),
]

# --- নির্ধারিত স্টাডি-টাইম উইন্ডো (মিনিটে) — স্ল্যাকিং-ডিটেকশনের কনটেক্সট-অ্যাওয়ারনেসের জন্য ---
MORNING_STUDY_START: int = 7 * 60          # 7:00 AM
MORNING_STUDY_END: int = 9 * 60 + 30       # 9:30 AM
NIGHT_STUDY_START: int = 19 * 60           # 7:00 PM
NIGHT_STUDY_END: int = 21 * 60             # 9:00 PM
