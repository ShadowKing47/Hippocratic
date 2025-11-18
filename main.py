# bedtime_story_system.py
import os
import openai
import random
import textwrap
import json
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()


"""
If I had 2 more hours, I would:
- Add a small persistent JSON "memory" that stores judge feedback and child preferences,
  and use that to adapt prompts over time (self-evolving prompt tuner).
- Implement a lightweight web UI (Flask) so a parent can type responses, choose mood,
  toggle surprise features, and play recommended soundtracks.
- Add unit tests and example fixtures for reproducible reviewer demos.
-Add character images and background images, to create picture book like characteristics
"""

# --- PLEASE DO NOT PUT YOUR OPENAI API KEY IN THIS FILE ---
# The script uses OPENAI_API_KEY from environment variables:
# export OPENAI_API_KEY="sk-..."  (do not commit this to git)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Keep this exactly as requested:
example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

# Themes and settings (cleaned, deduplicated, corrected)
THEMES = [
    "Adventure", "Animal Friends", "Problem Solving", "Friendship", "Magic/Fantasy",
    "Space", "Family", "Courage", "Kindness", "Honesty", "Justice", "Teamwork",
    "Be Yourself", "Love", "Hope", "Hard Work", "Loyalty", "Persistence",
    "Compassion", "Generosity", "Holidays", "Peace", "Equality", "Siblings",
    "Festival", "Summer Break", "Facing Fears", "Gratitude", "Self-Confidence",
    "Helping Others"
]

SETTINGS = [
    "Farm", "Ocean", "Jungle", "Forest", "Neighborhood", "Backyard", "Park",
    "Mountain", "Beach", "School", "Space", "Castle", "Under the Bed",
    "Treehouse", "Desert", "Arctic", "Magical Land", "Playground", "Circus",
    "Library"
]

MOOD_KEYWORDS = {
    "calm": ["calm", "sleepy", "tired", "cozy", "bedtime", "relaxed"],
    "anxious": ["scared", "lost", "anxious", "afraid", "worried", "nervous"],
    "excited": ["excited", "adventure", "energetic", "happy", "bouncy"],
    "sad": ["sad", "down", "lonely", "miss", "cry"],
    "curious": ["curious", "wonder", "ask", "how", "why", "what"],
    "playful": ["fun", "silly", "joke", "play"]
}

# Utility: call OpenAI ChatCompletion (gpt-3.5-turbo as required)
def call_model(prompt: str, max_tokens=700, temperature=0.6) -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Please set it before running.")
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message["content"]  # type: ignore

# Mini-psychologist: simple keyword-based mood detection (keeps privacy + offline-capable)
def detect_mood(user_text: str) -> str:
    lower = user_text.lower()
    scores = {k: 0 for k in MOOD_KEYWORDS.keys()}
    for mood, keys in MOOD_KEYWORDS.items():
        for kw in keys:
            if kw in lower:
                scores[mood] += 1
    # fallback: if user explicitly says mood
    for mood in ["calm", "anxious", "excited", "sad", "curious", "playful"]:
        if mood in lower:
            return mood
    # pick highest
    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        # default to calm for bedtime
        return "calm"
    return best[0]

# Theme selection: always include Problem Solving, plus 3 random distinct others
def select_themes() -> List[str]:
    others = [t for t in THEMES if t != "Problem Solving"]
    chosen = random.sample(others, k=3)
    chosen.append("Problem Solving")
    random.shuffle(chosen)
    return chosen

def select_setting() -> str:
    return random.choice(SETTINGS)

# Build the storyteller prompt with strong constraints suitable for ages 5-10
def build_story_prompt(user_request: str, mood: str, themes: List[str], setting: str) -> str:
    # tone constraints tuned for bedtime calm story requirements
    prompt = f"""
You are "StoryWeaver", an expert children's storyteller for ages 5 to 10.
Write a calm bedtime story based on the user's request and the generated context below.

USER REQUEST: {user_request}

CONTEXT:
- Mood detected: {mood}
- Themes (include all; problem solving must be implemented): {', '.join(themes)}
- Setting: {setting}

REQUIREMENTS (very important):
- Age: suitable for 5-10 year olds.
- Tone: soft rhythm, warm and calming.
- Imagery: bright, sensory, but gentle (colors, sounds, textures).
- Sentences: short and clear. Prefer sentences of 8-14 words.
- Emotions: clearly expressed and reassuring. Use simple labels ("Alice felt shy", "Bob was brave").
- Structure: simple arc with Introduction, Rising Action (include a small, solvable problem), Climax, Resolution, and a calming closing that encourages sleep.
- Length: 220-350 words.
- Dialogue: include 2-4 short lines of dialogue to increase engagement.
- Safety: no scary concepts, no violence, no adult themes.
- Language: simple vocabulary, avoid complex words; if a complex idea is needed, explain in one short sentence.

ADDITIONAL:
- Make "problem solving" explicit: show how characters discuss, try, and solve the small problem together.
- Reinforce calming ending: close with a sentence that helps the child relax for sleep (e.g., "Then they snuggled down and the stars hummed softly.").

Output ONLY the story text. Do not include analysis or extra metadata.
"""
    return prompt.strip()

# Judge prompt: evaluates story and returns JSON with scores, critique, and revision instructions
def build_judge_prompt(story_text: str) -> str:
    prompt = f"""
You are "StoryJudge", an expert evaluator of children's bedtime stories for ages 5-10.
Evaluate the story below in JSON format ONLY.

STORY:
\"\"\"{story_text}\"\"\"

EVALUATION CRITERIA (rate 1-10):
- AgeAppropriateness
- ToneCalmness
- ImageryQuality
- SentenceSimplicity
- EmotionalClarity
- ProblemSolvingPresence
- StructureCompleteness
- SleepinessFactor (how calming for bedtime)

Provide:
1) A JSON object with numeric scores for each criterion above.
2) A short textual critique (1-3 sentences).
3) Up to 5 concrete revision instructions to improve the story (each instruction short and actionable).

Output MUST be valid JSON only, with keys:
scores, critique, revisions

Example format:
{
  "scores": {"AgeAppropriateness":9, "ToneCalmness":8, ...},
  "critique": "Short critique here.",
  "revisions": ["Make the climax gentler.", "Add one line of dialogue.", ...]
}
"""
    return prompt.strip()

# Trading card generator prompt
def build_trading_cards_prompt(story_text: str) -> str:
    prompt = f"""
You are "CardMaker". From the story below, extract up to 3 main characters and produce three simple trading cards:
- Character card (name, 3 short traits, special little ability)
- Setting card (one card)
- Moral card (one card: one short sentence)
Format the output as JSON with keys: characters (list), setting, moral.
Keep each trait 1-3 words. Keep suitability for kids.
STORY:
\"\"\"{story_text}\"\"\"
"""
    return prompt.strip()

# Soundtrack recommendation prompt
def build_soundtrack_prompt(mood: str, setting: str) -> str:
    prompt = f"""
You are "SoundGuide". Recommend 3 short soundtrack suggestions to accompany a calm bedtime story.
Context:
- Mood: {mood}
- Setting: {setting}
For each suggestion include: title (short), short reason (1 sentence), and two example audio elements (e.g., "soft waves", "gentle harp").
Return JSON array of 3 objects.
"""
    return prompt.strip()

# Simple function to try parse judge JSON robustly
def parse_json_safe(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        # Attempt to extract first JSON-looking substring
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"error": "invalid_json", "raw": text}
        return {"error": "no_json", "raw": text}

# ASCII Block diagram printer
def print_block_diagram():
    diagram = r"""
Block Diagram (flow):
+----------------+      +-----------------+      +--------------+
|   User Input   | ---> | Mini-Psychologist| ---> | Categorizer  |
| (mood + request)|      +-----------------+      +--------------+
        |                        |                       |
        v                        v                       v
   +-------------------------------+    +----------------------------+
   |        Story Generator        |<---|   Memory / Prompt Tuner    |
   |         (StoryWeaver)         |    +----------------------------+
   +-------------------------------+
                 |
                 v
           +-------------+
           |  LLM Judge  |
           | (StoryJudge)|
           +------+------+  
                  |
          score >= thr? (yes)
             /           \
           yes           no
           /               \
  +----------------+     +------------------+
  | Final Story Out |     |  Rewriter (apply |
  +----------------+     |   judge fixes)   |
                         +------------------+
                                 |
                                 v
                         +------------------+
                         | Trading Cards +  |
                         | Soundtrack Engine|
                         +------------------+
"""
    print(diagram)


def generate_and_refine_story(user_input: str) -> Tuple[str, dict, dict, dict]:
    """
    - Generates story, runs judge, optionally rewrites up to 2 times.
    Returns (final_story, judge_json, cards_json, soundtrack_json)
    """
    # detect mood
    mood = detect_mood(user_input)
    themes = select_themes()
    setting = select_setting()

    # build and call storyteller
    story_prompt = build_story_prompt(user_input, mood, themes, setting)
    print("\n[DEBUG] Story prompt sent to storyteller (trimmed):\n", story_prompt[:800], "...\n")
    story = call_model(story_prompt, max_tokens=700, temperature=0.6)

    # judge loop
    for attempt in range(3):  # initial + up to 2 rewrites
        judge_prompt = build_judge_prompt(story)
        judge_raw = call_model(judge_prompt, max_tokens=500, temperature=0.3)
        judge_json = parse_json_safe(judge_raw)
        # if parse failed, create simple fallback judgement
        if "scores" not in judge_json:
            # safe fallback: compute naive scores (len-based)
            judge_json = {
                "scores": {
                    "AgeAppropriateness": 8,
                    "ToneCalmness": 7,
                    "ImageryQuality": 7,
                    "SentenceSimplicity": 7,
                    "EmotionalClarity": 7,
                    "ProblemSolvingPresence": 8,
                    "StructureCompleteness": 7,
                    "SleepinessFactor": 7
                },
                "critique": "Judge failed to return JSON; using fallback scores.",
                "revisions": ["Ensure the problem solving steps are explicit.", "Make ending more calming."]
            }

        # simple acceptance rule: average score >= 7.5
        scores = judge_json.get("scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        print(f"[DEBUG] Judge attempt {attempt+1} avg score: {avg_score:.2f}")
        if avg_score >= 7.5 or attempt == 2:
            # accept story
            break
        # else rewrite using judge revisions
        revisions = judge_json.get("revisions", [])[:5]
        # create rewrite prompt
        rewrite_instructions = " ".join(f"- {r}" for r in revisions)
        rewrite_prompt = f"""
You are StoryWeaver. Please rewrite the story below applying these revision instructions:
{rewrite_instructions}

Original story:
\"\"\"{story}\"\"\"

Constraints (keep these!):
- Maintain age-appropriateness (5-10)
- Keep tone calming and the ending sleep-friendly
- Preserve characters and moral; emphasize problem-solving steps more clearly
- Keep sentences short (8-14 words)
- Length between 220-350 words

Output only the rewritten story text.
"""
        print("[DEBUG] Sending rewrite prompt to storyteller.")
        story = call_model(rewrite_prompt, max_tokens=700, temperature=0.55)

    # generate trading cards
    cards_prompt = build_trading_cards_prompt(story)
    cards_raw = call_model(cards_prompt, max_tokens=300, temperature=0.3)
    try:
        cards_json = json.loads(cards_raw)
    except Exception:
        # attempt to extract JSON or fallback
        cards_json = parse_json_safe(cards_raw)

    # generate soundtrack recommendations
    sound_prompt = build_soundtrack_prompt(mood, setting)
    sound_raw = call_model(sound_prompt, max_tokens=300, temperature=0.3)
    try:
        soundtrack_json = json.loads(sound_raw)
    except Exception:
        soundtrack_json = parse_json_safe(sound_raw)

    return story, judge_json, cards_json, soundtrack_json, mood, themes, setting

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

def main():
    print("Example request (kept for reference):")
    print(example_requests)
    print("\nEnter your child's request or mood hint. Examples:")
    print("- 'I want a story about Alice and Bob the cat'")
    print("- 'sleepy and calm, story about a shy turtle'")
    print("- Press ENTER to use the example request\n")

    user_input = input("What kind of story do you want to hear? ").strip()

    # If user does not type anything, use the example request
    if not user_input:
        user_input = example_requests
        print(f"\nUsing default example request:\n{user_input}\n")

    print_block_diagram()
    print("\nGenerating story. Please wait...\n")

    try:
        story, judge_json, cards_json, soundtrack_json, mood, themes, setting = generate_and_refine_story(user_input)
    except Exception as e:
        print("Error while generating story:", e)
        return

    # output results
    print("\n" + "="*60)
    print("Detected mood:", mood)
    print("Selected setting:", setting)
    print("Selected themes (4):", ", ".join(themes))
    print("="*60)
    print("\nFINAL STORY:\n")
    print(textwrap.fill(story, width=80))
    print("\n" + "-"*60)
    print("JUDGE FEEDBACK (raw):")
    print(json.dumps(judge_json, indent=2))
    print("\nTRADING CARDS (parsed):")
    print(json.dumps(cards_json, indent=2))
    print("\nSOUNDTRACK SUGGESTIONS (parsed):")
    print(json.dumps(soundtrack_json, indent=2))
    print("\n" + "="*60)
    print("Notes:")
    print("- If you want to tweak tone or length, re-run and add instructions (e.g., 'make it even gentler').")
    print("- Store judge feedback in a JSON file to enable the self-evolving tuning feature.")

if __name__ == "__main__":
    main()
