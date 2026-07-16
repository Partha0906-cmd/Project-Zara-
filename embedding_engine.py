"""
embedding_engine.py
প্রজেক্ট জারা - Word Embedding ভিত্তিক Semantic Matching (Elite-tier আপগ্রেড, ধাপ ৪)।

TF-IDF (intent_engine.py-এর Layer 2) শব্দের "উপস্থিতি" দিয়ে মিল বোঝে — কিন্তু "ক্লান্ত" আর
"টায়ার্ড" যে প্রায় একই অর্থ বহন করে, সেটা TF-IDF জানে না (এগুলো ভিন্ন string)। এই মডিউল
Facebook-এর fastText থেকে prune করা Bengali word vectors ব্যবহার করে প্রকৃত অর্থগত
কাছাকাছি-তা মাপে — সম্পূর্ণ অফলাইন, কোনো numpy/sklearn ছাড়াই (শুধু Python-এর built-in
`array` মডিউল, RAM-এ কমপ্যাক্ট রাখতে)।

ফাইল ফরম্যাট (config.BENGALI_EMBEDDINGS_FILE):
    প্রতি লাইনে: "শব্দ<TAB>v1,v2,...,v100"  (100-dim, মূল fastText 300-dim থেকে truncate করা)

এই ফাইলটা ঐচ্ছিক (optional) — না পাওয়া গেলে বা করাপ্ট হলে embedding layer নিষ্ক্রিয় থাকে,
প্রোগ্রাম স্বাভাবিকভাবেই চলবে (শুধু intent_engine-এর TF-IDF fallback দিয়েই matching হবে)।
"""

import math
import os
from array import array
from typing import Dict, List, Optional

from config import BENGALI_EMBEDDINGS_FILE

_EMBEDDING_DIM = 100

# module-level cache — একবারই লোড হয়
_WORD_VECTORS: Dict[str, array] = {}
_LOAD_ATTEMPTED = False


def _load_embeddings() -> None:
    """
    config.BENGALI_EMBEDDINGS_FILE থেকে word vectors লোড করে _WORD_VECTORS-এ ক্যাশ করে।
    ফাইল অনুপস্থিত/করাপ্ট হলে চুপচাপ খালি dict রেখে দেয় (ক্র্যাশ করে না) — is_available()
    দিয়ে ইউজার/অন্য মডিউল বুঝতে পারবে embedding layer সক্রিয় আছে কিনা।
    """
    global _LOAD_ATTEMPTED
    _LOAD_ATTEMPTED = True

    if not os.path.exists(BENGALI_EMBEDDINGS_FILE):
        return

    try:
        with open(BENGALI_EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                word, vec_str = line.split("\t", 1)
                try:
                    values = array("f", (float(v) for v in vec_str.split(",")))
                except ValueError:
                    continue
                if len(values) == _EMBEDDING_DIM:
                    _WORD_VECTORS[word] = values
    except (OSError, UnicodeDecodeError) as e:
        print(f"[সিস্টেম সতর্কতা] Bengali word embeddings লোড করতে সমস্যা হয়েছে ({e}). এই লেয়ার ছাড়াই চলবে।")
        _WORD_VECTORS.clear()


def is_available() -> bool:
    """Embedding layer আদৌ সক্রিয় আছে কিনা (ফাইল ঠিকভাবে লোড হয়েছে কিনা) জানায়।"""
    if not _LOAD_ATTEMPTED:
        _load_embeddings()
    return len(_WORD_VECTORS) > 0


def get_vector(word: str) -> Optional[array]:
    """একটা শব্দের ভেক্টর রিটার্ন করে, vocabulary-তে না থাকলে None।"""
    if not _LOAD_ATTEMPTED:
        _load_embeddings()
    return _WORD_VECTORS.get(word)


def _cosine_similarity(vec_a: array, vec_b: array) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def word_similarity(word_a: str, word_b: str) -> float:
    """দুটো শব্দের অর্থগত কাছাকাছি-তা (0.0–1.0) মাপে। কোনোটা vocabulary-তে না থাকলে 0.0।"""
    vec_a = get_vector(word_a)
    vec_b = get_vector(word_b)
    if vec_a is None or vec_b is None:
        return 0.0
    return _cosine_similarity(vec_a, vec_b)


def sentence_vector(words: List[str]) -> Optional[array]:
    """
    একগুচ্ছ শব্দের গড় ভেক্টর (centroid) বানায় — vocabulary-তে না থাকা শব্দ উপেক্ষা করা হয়।
    একটাও শব্দ vocabulary-তে না থাকলে None রিটার্ন করে।
    """
    vectors = [get_vector(w) for w in words]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    summed = array("f", [0.0] * _EMBEDDING_DIM)
    for vec in vectors:
        for i in range(_EMBEDDING_DIM):
            summed[i] += vec[i]
    count = len(vectors)
    return array("f", (v / count for v in summed))


def weighted_sentence_vector(words: List[str], weights: Dict[str, float]) -> Optional[array]:
    """
    IDF-ওয়েটেড গড় ভেক্টর বানায় — প্রতিটা শব্দের অবদান তার IDF-ওয়েট অনুযায়ী স্কেল হয়।
    এটা জরুরি: simple (unweighted) averaging-এ "আজকে"/"খুব"-এর মতো সাধারণ শব্দ, যেগুলো
    প্রায় প্রতিটা ইনটেন্টের অনেক phrase-এ বারবার আসে, distinctive content word (যেমন
    "ফেসবুক"/"রিলস")-কে ছাপিয়ে centroid-কে ভুল দিকে টেনে নিতে পারে। weights (সাধারণত
    intent_engine._IDF থেকে) দিয়ে সাধারণ শব্দের প্রভাব কমানো হয়।
    vocabulary-তে না থাকা শব্দ, বা weight পাওয়া যায়নি এমন শব্দ, উপেক্ষা করা হয়।
    """
    weighted_pairs = []
    for w in words:
        vec = get_vector(w)
        if vec is not None:
            weight = weights.get(w, 1.0)
            weighted_pairs.append((vec, weight))

    if not weighted_pairs:
        return None

    total_weight = sum(w for _, w in weighted_pairs)
    if total_weight == 0:
        return None

    summed = array("f", [0.0] * _EMBEDDING_DIM)
    for vec, weight in weighted_pairs:
        for i in range(_EMBEDDING_DIM):
            summed[i] += vec[i] * weight
    return array("f", (v / total_weight for v in summed))
