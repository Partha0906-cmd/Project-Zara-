Project Zara
Overview
Project Zara is a personal AI room assistant designed to become an
always-on smart companion.
Vision
Natural conversation
Smart room control
Personal memory
Automation
Low-cost hardware
AI Team
Role           Member                Responsibility
Project Lead   Nick Fury (Partho)    Decisions and planning
Architect      Thor (ChatGPT)        Ideas and architecture
Engineer       Tony Stark (Claude)   Coding and debugging
Reviewer       Steve Rogers (Grok)   Review and optimization
Hardware
Raspberry Pi Zero 2 W (chosen — cost-optimized)
Main AI brain
Runs the offline intent engine + memory system
Same core capability as Pi 4/5 for this workload, much lower cost
Upgrade path: Pi 4/5 only needed later if we add a local LLM (RAM-heavy)
ESP32
Relay control
Sensor management
Device communication
Optional Devices
Microphone
Speaker
ESP32-CAM
API Stack (Planned — Not Yet Integrated)
Current system is 100% offline, no API calls anywhere. These are future
"Online Mode" candidates from the original blueprint, to be added later:
Gemini API (real-time conversational fallback)
Groq API (fast inference alternative)
Weather API
Alpha Vantage API (financial analytics hub, low priority)
Offline Intent Engine (Built — Core Achievement So Far)
This is the actual working software, fully offline, no internet or API
required. It runs as a 4-layer fallback pipeline in intent_engine.py:
Layer                Purpose                                          File
Keyword Matching   Exact + fuzzy (typo-tolerant) phrase matching    intent_engine.py
TF-IDF Semantic    Catches new phrasing sharing no exact keywords   intent_engine.py
Word Embeddings    Catches synonyms (e.g. "ক্লান্ত" ~ "দুর্বল")     embedding_engine.py
Self-Learning Log  Anything still unmatched gets logged for review learning_engine.py
Extra subsystems built on top of the intent engine:
Negation handling — "ক্লান্ত না" correctly does not trigger the
"tired" intent (with a safety exception for built-in idioms like
"ভালো লাগছে না").
Active Self-Learning ("শেখাও") — user can interactively teach Zara
new phrases at runtime; they persist to learned_keywords.json and
take effect immediately, no restart or code edit needed.
Short-term Conversation Context (context_engine.py) — remembers
the last 3 turns for:
Pronoun resolution ("এটা মনে রাখো" → saves what was just said)
Repeat command ("ওটা আবার বলো" → repeats last reply)
Follow-up awareness ("কতক্ষণ?" after "আজকে ক্লান্ত লাগছে" →
acknowledges the "বিশ্রাম/ক্লান্তি" topic instead of a generic
"didn't understand" reply)
Word vectors — pruned Bengali fastText vectors
(bn_vectors_pruned.txt, ~15,000 words, 100-dim, ~11MB), IDF-weighted
and stopword-filtered to avoid generic words skewing similarity.
All of this runs on pure Python (re, math, json, array,
collections) — no numpy, no sklearn, no pip installs needed, so it
drops straight into Pydroid3 or the Pi with zero extra setup.
Repository Files
Code (Python modules):
config.py
context_engine.py
embedding_engine.py
intent_engine.py
learning_engine.py
main.py
memory_engine.py
utils.py
Data (runtime-generated / bundled, not hand-written code):
zara_memory.json — permanent saved-memory storage
unmatched_log.json — self-learning log of not-yet-understood inputs
learned_keywords.json — phrases taught via "শেখাও" (created on first use)
bn_vectors_pruned.txt — pruned Bengali word vectors (~11MB, static)
Roadmap
Phase 1 — ✅ Done
Offline AI brain (4-layer intent engine: keyword, TF-IDF, embeddings,
self-learning)
Permanent memory (save/recall/search/delete, with confirmation flow)
Short-term conversation context (pronoun resolution, repeat, follow-up)
Active self-learning ("শেখাও" — teaches new phrases at runtime)
Phase 2 — Not Started
Voice input/output (Vosk STT + Piper TTS)
Wake word ("হে জারা") via always-listening mic loop
ESP32 integration (GPIO, relay control)
Phase 3
Automation (lights/fan control)
Camera + motion detection
Scheduling refinements
Phase 4
Multi-room support
Advanced learning (possible local LLM upgrade path, if hardware allows)
Principles
Modular
Low cost
Expandable
Offline-first where possible
License
Personal learning and research project.
