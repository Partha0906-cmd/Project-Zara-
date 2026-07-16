"""
utils.py
প্রজেক্ট জারা - হেল্পার ইউটিলিটি মডিউল।
সময়-ভিত্তিক ক্যালকুলেশন (স্কুল লক, টিউশন স্লট, রুটিন চেক) এখানে থাকে।
"""

import datetime
from typing import NamedTuple

import config
from config import SCHOOL_START, SCHOOL_END, SATURDAY_SCHOOL_END, REFRESH_END, TUITION_END, TUITION_DAYS


class TimeContext(NamedTuple):
    """বর্তমান সময়-সম্পর্কিত সমস্ত তথ্য একত্রে বহন করার জন্য একটি ইমিউটেবল স্ট্রাকচার।"""
    current_time: datetime.datetime
    hour: int
    minute: int
    day_index: int
    date_str: str
    current_mins: int
    is_tuition_day: bool


def get_time_context() -> TimeContext:
    """বর্তমান সময় থেকে প্রয়োজনীয় সমস্ত ডেরাইভড তথ্য (মিনিট, তারিখ, টিউশন ডে কিনা) হিসাব করে রিটার্ন করে।"""
    current_time = datetime.datetime.now()
    hour = current_time.hour
    minute = current_time.minute
    day_index = current_time.weekday()  # 0 = Monday ... 6 = Sunday
    date_str = current_time.strftime("%Y-%m-%d")
    current_mins = hour * 60 + minute
    is_tuition_day = day_index in TUITION_DAYS

    return TimeContext(
        current_time=current_time,
        hour=hour,
        minute=minute,
        day_index=day_index,
        date_str=date_str,
        current_mins=current_mins,
        is_tuition_day=is_tuition_day,
    )


def is_school_time(current_mins: int, day_index: int) -> bool:
    """
    বর্তমান সময় স্কুল লক জোনের মধ্যে আছে কিনা তা যাচাই করে।
    রবিবার (6): স্কুল সবসময় বন্ধ, তাই কখনোই লক হবে না।
    শনিবার (5): স্কুল শর্ট ডে, ১০:৪০ AM - ২:০০ PM পর্যন্ত লক।
    অন্যান্য দিন: স্বাভাবিক ১০:৪০ AM - ৪:৩০ PM লক।
    """
    if day_index == 6:
        return False
    school_end = SATURDAY_SCHOOL_END if day_index == 5 else SCHOOL_END
    return SCHOOL_START <= current_mins < school_end


def is_refresh_time(current_mins: int, day_index: int) -> bool:
    """
    বর্তমান সময় পোস্ট-স্কুল রিফ্রেশ স্লটের মধ্যে আছে কিনা তা যাচাই করে।
    শনি/রবি-তে আলাদা শিডিউল (Rest/My Time) প্রযোজ্য, তাই এই নিয়মিত রিফ্রেশ-স্লট এই দুই দিনে প্রযোজ্য নয়।
    """
    if day_index in (5, 6):
        return False
    return SCHOOL_END <= current_mins < REFRESH_END


def is_tuition_time(current_mins: int, is_tuition_day: bool) -> bool:
    """বর্তমান সময় টিউশন স্লটের মধ্যে এবং আজ টিউশন ডে কিনা তা যাচাই করে।"""
    return is_tuition_day and (REFRESH_END <= current_mins < TUITION_END)


def format_time_12hr(current_time: datetime.datetime) -> str:
    """datetime অবজেক্টকে 12-ঘণ্টা ফরম্যাটের স্ট্রিং-এ (যেমন 05:30 PM) রূপান্তর করে।"""
    return current_time.strftime("%I:%M %p")


def get_day_type_label(day_index: int) -> str:
    """দিনের ইনডেক্স অনুযায়ী 'লার্নিং ডে' বা 'প্রোডাক্টিভ ডে' লেবেল রিটার্ন করে।"""
    if day_index in config.TUITION_DAYS:
        return config.LEARNING_DAY_LABEL
    return config.PRODUCTIVE_DAY_LABEL


def get_schedule_for_day(day_index: int) -> list:
    """
    দিনের ইনডেক্স অনুযায়ী সঠিক শিডিউল (সময়-ব্লক লিস্ট) রিটার্ন করে।
    শনিবার (5) ও রবিবার (6) এর জন্য আলাদা ব্যতিক্রমী শিডিউল আছে।
    """
    if day_index == 5:  # শনিবার
        return config.SCHEDULE_SATURDAY
    if day_index == 6:  # রবিবার
        return config.SCHEDULE_SUNDAY
    if day_index in config.TUITION_DAYS:  # মঙ্গল, বৃহস্পতি — সাধারণ লার্নিং ডে
        return config.SCHEDULE_LEARNING_DAY
    return config.SCHEDULE_PRODUCTIVE_DAY  # সোম, বুধ, শুক্র


def is_study_time(current_mins: int) -> bool:
    """বর্তমান সময় নির্ধারিত সেলফ-স্টাডি (সকাল) বা নাইট-স্টাডি স্লটের মধ্যে আছে কিনা যাচাই করে।"""
    in_morning_study = config.MORNING_STUDY_START <= current_mins < config.MORNING_STUDY_END
    in_night_study = config.NIGHT_STUDY_START <= current_mins < config.NIGHT_STUDY_END
    return in_morning_study or in_night_study
