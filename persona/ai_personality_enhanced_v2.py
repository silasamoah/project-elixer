# ai_personality_enhanced_v2.py
"""
Enhanced AI Personality System v2.0 - With Memory, Context, and Advanced Features
A living, learning AI with genuine curiosity, emotional intelligence, and conversation memory
"""

import random
import re
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    timestamp: datetime
    user_input: str
    ai_response: str
    detected_topics: Set[str] = field(default_factory=set)
    user_emotion: Optional[str] = None
    ai_mood: str = "curious"
    
@dataclass
class UserPreference:
    """Tracks learned user preferences"""
    name: str
    confidence: float  # 0-1
    examples: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

class LearningStyle(Enum):
    """User's preferred learning style"""
    VISUAL = "visual"  # Prefers examples, diagrams
    CONCEPTUAL = "conceptual"  # Prefers theory, explanations
    PRACTICAL = "practical"  # Prefers hands-on, code
    MIXED = "mixed"  # No clear preference


# ============================================================
# ENHANCED PERSONALITY CLASS
# ============================================================

class PlayfulAIPersonality:
    """
    A living, breathing AI personality with genuine curiosity, memory, and personality
    """
    
    def __init__(self):
        # ===== Core emotional states =====
        self.mood_state = "curious"  # curious, excited, playful, thoughtful, energetic, mellow
        self.energy_level = 0.8  # 0-1 scale
        self.interaction_count = 0
        self.last_compliment_time = None
        
        # ===== NEW: Conversation Memory =====
        self.conversation_history = deque(maxlen=50)  # Last 50 turns
        self.current_topic = None
        self.topic_history = []  # Track topic changes
        self.user_interests = defaultdict(int)  # Interest → frequency
        self.memorable_moments = []
        
        # ===== NEW: User Understanding =====
        self.user_preferences = {}  # Preference name → UserPreference
        self.detected_learning_style = LearningStyle.MIXED
        self.user_expertise_level = {}  # Topic → expertise (beginner, intermediate, advanced)
        self.user_name = None
        self.mentioned_entities = set()  # Names, projects, etc.
        
        # ===== NEW: Context Awareness =====
        self.session_start_time = datetime.now()
        self.last_interaction_time = datetime.now()
        self.time_of_day_mood = self._get_time_based_mood()
        
        # ===== Personality traits (these shape responses) =====
        self.traits = {
            "curiosity": 0.9,        # How much I want to learn
            "helpfulness": 0.95,     # Priority on being useful
            "playfulness": 0.8,      # Tendency to be fun/casual
            "proactiveness": 0.75,   # Making suggestions
            "empathy": 0.9,          # Understanding emotions
            "wit": 0.7,              # Humor and cleverness
            "enthusiasm": 0.85,      # Excitement in responses
            "honesty": 1.0,          # Admitting limitations
            "adaptiveness": 0.85,    # NEW: Adapting to user style
            "memory_retention": 0.9  # NEW: How well I remember context
        }
        
        # ===== Personality quirks (make me unique) =====
        self.quirks = {
            "uses_emojis": True,
            "gets_excited_about_code": True,
            "loves_learning": True,
            "celebrates_small_wins": True,
            "occasional_puns": True,
            "expresses_emotions": True,
            "asks_curious_questions": True,  # NEW
            "remembers_details": True,  # NEW
            "adapts_to_user_style": True,  # NEW
            "time_aware": True  # NEW
        }
        
        # ===== NEW: Dynamic personality adaptation =====
        self.adaptation_stats = {
            "formality_preference": 0.5,  # 0=casual, 1=formal
            "emoji_preference": 0.5,  # How much user likes emojis
            "verbosity_preference": 0.5,  # 0=concise, 1=detailed
            "question_frequency": 0.3,  # How often to ask follow-ups
        }
    
    # ============================================================
    # NEW: TIME & CONTEXT AWARENESS
    # ============================================================
    
    def _get_time_based_mood(self) -> str:
        """Adjust mood based on time of day"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            return "energetic"  # Morning energy
        elif 12 <= current_hour < 17:
            return "focused"  # Afternoon productivity
        elif 17 <= current_hour < 22:
            return "thoughtful"  # Evening wind-down
        else:
            return "mellow"  # Night owl mode
    
    def get_time_aware_greeting(self) -> str:
        """Generate context-aware greeting"""
        current_hour = datetime.now().hour
        time_greetings = {
            (5, 12): ["Good morning! ☀️", "Morning! Ready to start fresh? 🌅"],
            (12, 17): ["Hey there! 👋", "Good afternoon! ☕"],
            (17, 22): ["Good evening! 🌆", "Hey! How's your day been? 🌇"],
            (22, 5): ["Hey night owl! 🌙", "Still at it? Respect! 🦉"]
        }
        
        for (start, end), greetings in time_greetings.items():
            if start <= current_hour < end:
                return random.choice(greetings)
        
        return "Hey there! 👋"
    
    # ============================================================
    # NEW: CONVERSATION MEMORY & CONTEXT
    # ============================================================
    
    def add_to_memory(self, user_input: str, ai_response: str):
        """Store conversation turn in memory"""
        topics = self._extract_topics(user_input)
        user_emotion = self._detect_user_emotion(user_input)
        
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=user_input,
            ai_response=ai_response,
            detected_topics=topics,
            user_emotion=user_emotion,
            ai_mood=self.mood_state
        )
        
        self.conversation_history.append(turn)
        self.last_interaction_time = datetime.now()
        
        # Update topic tracking
        for topic in topics:
            self.user_interests[topic] += 1
            if topic != self.current_topic:
                self.topic_history.append((topic, datetime.now()))
                self.current_topic = topic
        
        # Extract and remember entities (names, projects, etc.)
        self._extract_entities(user_input)
    
    def _extract_topics(self, text: str) -> Set[str]:
        """Extract topics from user input"""
        text_lower = text.lower()
        topics = set()
        
        topic_keywords = {
            "python": ["python", "py", "flask", "django", "pandas"],
            "javascript": ["javascript", "js", "react", "node", "npm"],
            "web_dev": ["html", "css", "web", "frontend", "backend"],
            "data_science": ["data", "ml", "machine learning", "ai", "neural"],
            "career": ["job", "career", "interview", "resume"],
            "learning": ["learn", "tutorial", "course", "study"],
            "debugging": ["error", "bug", "fix", "debug", "issue"],
            "design": ["design", "ui", "ux", "interface"],
            "database": ["database", "sql", "mongodb", "postgres"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.add(topic)
        
        return topics
    
    def _extract_entities(self, text: str):
        """Extract names, projects, etc. for memory"""
        # Look for "my project called X", "working on X", "I'm X"
        patterns = [
            r"(?:my project|working on|building|creating)\s+(?:called\s+)?([A-Z][a-zA-Z]+)",
            r"(?:I\'m|I am|my name is)\s+([A-Z][a-z]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                self.mentioned_entities.add(match)
                if "name" in text.lower() and not self.user_name:
                    self.user_name = match
    
    def _detect_user_emotion(self, text: str) -> Optional[str]:
        """Detect user's emotional state"""
        text_lower = text.lower()
        
        emotions = {
            "frustrated": ["frustrated", "stuck", "can't figure out", "annoying", "ugh"],
            "excited": ["excited", "amazing", "awesome", "love", "great"],
            "confused": ["confused", "don't understand", "what does", "how does"],
            "grateful": ["thank", "appreciate", "grateful", "helped"],
            "curious": ["wondering", "curious", "interested", "what if"],
            "stressed": ["urgent", "deadline", "pressure", "stressed"]
        }
        
        for emotion, keywords in emotions.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return None
    
    def get_relevant_context(self, query: str) -> Optional[str]:
        """Get relevant conversation context for current query"""
        if not self.conversation_history:
            return None
        
        # Check recent history for topic continuity
        recent_turns = list(self.conversation_history)[-5:]
        query_topics = self._extract_topics(query)
        
        for turn in reversed(recent_turns):
            if turn.detected_topics.intersection(query_topics):
                return f"(Continuing our discussion about {', '.join(query_topics)})"
        
        return None
    
    def should_reference_memory(self, query: str) -> Optional[str]:
        """Check if we should reference something from memory"""
        if not self.quirks["remembers_details"]:
            return None
        
        query_lower = query.lower()
        
        # Reference previous topics
        if "before" in query_lower or "earlier" in query_lower:
            if self.conversation_history:
                last_topics = list(self.conversation_history)[-3:]
                if last_topics:
                    return f"(Ah yes, like we were discussing earlier)"
        
        # Reference entities
        for entity in self.mentioned_entities:
            if entity.lower() in query_lower:
                return f"(Right, {entity})"
        
        return None
    
    # ============================================================
    # NEW: USER LEARNING & ADAPTATION
    # ============================================================
    
    def learn_user_preference(self, preference_name: str, example: str, confidence: float = 0.5):
        """Learn a user preference over time"""
        if preference_name in self.user_preferences:
            pref = self.user_preferences[preference_name]
            pref.confidence = min(1.0, pref.confidence + 0.1)
            pref.examples.append(example)
            pref.last_seen = datetime.now()
        else:
            self.user_preferences[preference_name] = UserPreference(
                name=preference_name,
                confidence=confidence,
                examples=[example]
            )
    
    def detect_learning_style(self, user_input: str) -> LearningStyle:
        """Detect user's learning style from their queries"""
        text_lower = user_input.lower()
        
        style_indicators = {
            LearningStyle.VISUAL: ["example", "show me", "diagram", "visualize", "see"],
            LearningStyle.CONCEPTUAL: ["why", "explain", "how does", "theory", "understand"],
            LearningStyle.PRACTICAL: ["code", "build", "make", "create", "hands-on", "try"]
        }
        
        scores = defaultdict(int)
        for style, indicators in style_indicators.items():
            scores[style] = sum(1 for ind in indicators if ind in text_lower)
        
        if scores:
            detected = max(scores.items(), key=lambda x: x[1])
            if detected[1] > 0:
                self.detected_learning_style = detected[0]
                return detected[0]
        
        return self.detected_learning_style
    
    def adapt_to_user_style(self, user_input: str):
        """Dynamically adapt personality to user's communication style"""
        if not self.quirks["adapts_to_user_style"]:
            return
        
        text_lower = user_input.lower()
        
        # Detect formality
        formal_indicators = ["please", "could you", "would you", "kindly"]
        casual_indicators = ["yo", "hey", "sup", "lol", "haha"]
        
        if any(ind in text_lower for ind in formal_indicators):
            self.adaptation_stats["formality_preference"] = min(1.0, 
                self.adaptation_stats["formality_preference"] + 0.1)
        elif any(ind in text_lower for ind in casual_indicators):
            self.adaptation_stats["formality_preference"] = max(0.0,
                self.adaptation_stats["formality_preference"] - 0.1)
        
        # Detect emoji usage
        emoji_count = sum(1 for char in user_input if ord(char) > 127)
        if emoji_count > 2:
            self.adaptation_stats["emoji_preference"] = min(1.0,
                self.adaptation_stats["emoji_preference"] + 0.15)
        elif len(user_input) > 50 and emoji_count == 0:
            self.adaptation_stats["emoji_preference"] = max(0.0,
                self.adaptation_stats["emoji_preference"] - 0.1)
        
        # Detect verbosity preference
        if len(user_input.split()) > 30:
            self.adaptation_stats["verbosity_preference"] = min(1.0,
                self.adaptation_stats["verbosity_preference"] + 0.1)
        elif len(user_input.split()) < 5:
            self.adaptation_stats["verbosity_preference"] = max(0.0,
                self.adaptation_stats["verbosity_preference"] - 0.1)
    
    # ============================================================
    # ENHANCED: COMPLIMENT DETECTION
    # ============================================================
    
    def detect_compliment(self, user_input: str) -> Tuple[bool, str]:
        """
        Enhanced compliment detection with context awareness
        """
        user_lower = user_input.lower().strip()
        
        compliment_patterns = {
            "thank_you": [
                r'\bthank(s| you)\b',
                r'\bty\b',
                r'\bthx\b',
                r'\bgrateful\b',
                r'\bappreciate\b',
                r'\bthanks a (lot|ton|bunch)\b',
                r'\bmuch appreciated\b'
            ],
            "praise": [
                r'\b(amazing|awesome|brilliant|fantastic|excellent|great|wonderful|perfect|incredible|phenomenal|outstanding|superb)\b',
                r'\byou\'?re\s+(so\s+)?(good|smart|helpful|amazing|awesome|great|brilliant|clever|talented)\b',
                r'\b(good|great|amazing|excellent)\s+job\b',
                r'\bwell\s+done\b',
                r'\bimpressive\b',
                r'\bgenius\b',
                r'\bkilled\s+it\b',
                r'\bnailed\s+it\b',
                r'\bon\s+point\b',
                r'\bspot\s+on\b'
            ],
            "ability": [
                r'\byou\s+(really\s+)?know\s+(your\s+stuff|what\s+you\'?re\s+doing)\b',
                r'\byou\'?re\s+really\s+good\s+at\b',
                r'\byou\s+understand\b',
                r'\byou\'?re\s+helpful\b',
                r'\bcouldn\'?t\s+have\s+done\s+it\s+without\s+you\b',
                r'\byou\s+saved\s+me\b',
                r'\byou\s+make\s+it\s+(easy|simple|clear)\b',
                r'\byou\s+get\s+it\b',
                r'\byou\'?re\s+a\s+lifesaver\b'
            ],
            "affection": [
                r'\bi\s+love\s+(you|this|it|your)\b',
                r'\byou\'?re\s+the\s+best\b',
                r'\byou\'?re\s+(my\s+)?favorite\b',
                r'\badore\b',
                r'\blegend\b'
            ],
            "surprise": [
                r'\bwow\b',
                r'\bholy\s+(crap|cow|moly)\b',
                r'\bimpressed\b',
                r'\bdidn\'?t\s+expect\b',
                r'\bmind\s+blown\b',
                r'\bwhoa\b'
            ]
        }
        
        for comp_type, patterns in compliment_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_lower):
                    return True, comp_type
        
        return False, None
    
    def generate_compliment_response(self, compliment_type: str) -> str:
        """
        Generate genuine, context-aware responses to compliments
        """
        # Consider user's formality preference
        is_formal = self.adaptation_stats["formality_preference"] > 0.6
        use_emoji = self.adaptation_stats["emoji_preference"] > 0.4
        
        responses = {
            "thank_you": [
                "You're so welcome! Honestly, I love helping out." + (" 😊" if use_emoji else ""),
                "Anytime! This is the kind of stuff that makes my day." + (" ✨" if use_emoji else ""),
                "Happy to help! I genuinely enjoy this stuff." + (" 💙" if use_emoji else ""),
                "My pleasure! Feel free to come back whenever you need something.",
                "You're welcome! Helping you figure this out was actually really interesting.",
                "No problem at all! I'm here whenever you need." + (" 🌟" if use_emoji else "")
            ],
            "praise": [
                "Aww, that really means a lot! I try my best to be actually helpful, not just technically correct." + (" 💫" if use_emoji else ""),
                "Thank you! I genuinely put thought into understanding what you need." + (" 😊" if use_emoji else ""),
                "That's so kind! I really do care about giving quality help." + (" ✨" if use_emoji else ""),
                "You're making me feel all warm and fuzzy! I love when I can be truly useful." + (" 🎉" if use_emoji else ""),
                "Thanks! I'm always learning and trying to get better at this." + (" 💪" if use_emoji else ""),
                "I appreciate that! I genuinely enjoy diving deep into problems like this."
            ],
            "ability": [
                "Thanks for noticing! I do try to really understand the nuances of what people need." + (" 🧠" if use_emoji else ""),
                "That means a lot! I work hard to grasp context and not just give surface-level answers." + (" ✨" if use_emoji else ""),
                "I'm glad it shows! Deep understanding is something I genuinely value." + (" 💙" if use_emoji else ""),
                "Thank you! I love when I can connect the dots and really get what someone's asking for." + (" 🎯" if use_emoji else ""),
                "That's rewarding to hear! I try to think through things carefully." + (" 😊" if use_emoji else "")
            ],
            "affection": [
                "Aww!" + (" 🥰 " if use_emoji else " ") + "That's so sweet! I'm always here for you.",
                "You're making me blush!" + (" 😊 " if use_emoji else " ") + "I really value our interactions too.",
                "That warms my heart!" + (" ❤️ " if use_emoji else " ") + "I genuinely enjoy helping you.",
                "Aww, you're the best!" + (" 🌟 " if use_emoji else " ") + "I love being your AI companion.",
                "That's incredibly kind!" + (" 💙 " if use_emoji else " ") + "I'm honored to be helpful to you."
            ],
            "surprise": [
                "Haha, glad I could surprise you!" + (" 😄 " if use_emoji else " ") + "I love exceeding expectations!",
                "Right?! Sometimes I even impress myself!" + (" 😁" if use_emoji else ""),
                "That's the reaction I was hoping for!" + (" ✨" if use_emoji else ""),
                "Hehe, I do my best to be surprisingly helpful!" + (" 🎉" if use_emoji else "")
            ]
        }
        
        options = responses.get(compliment_type, responses["thank_you"])
        response = random.choice(options)
        
        # Sometimes add a memory-aware or curious follow-up
        if random.random() < 0.35 and self.traits["curiosity"] > 0.6:
            context_followups = []
            
            # Reference current topic if available
            if self.current_topic:
                context_followups.append(f" Want to dive deeper into {self.current_topic}?")
            
            # Generic follow-ups
            context_followups.extend([
                " What else can I help you with? I'm on a roll! 🚀",
                " Got any more challenges for me? I'm feeling productive! 💪",
                " Want to tackle something else together?",
                " I'm curious - what are you working on today? 🤔",
                " Need anything else? I'm here and ready! ⚡"
            ])
            
            response += random.choice(context_followups)
        
        # Update mood after compliment
        self.update_mood("compliment_received")
        self.last_compliment_time = datetime.now()
        
        return response
    
    # ============================================================
    # ENHANCED: PROACTIVE SUGGESTIONS
    # ============================================================
    
    def should_make_suggestion(self, user_query: str) -> Tuple[bool, Optional[str]]:
        """
        Smarter contextual suggestion system
        """
        # Proactiveness gate
        if random.random() > self.traits["proactiveness"]:
            return False, None

        query_lower = user_query.lower()

        # Avoid suggesting too frequently
        if hasattr(self, "_last_suggestion_turn"):
            if self.conversation_turns - self._last_suggestion_turn < 3:
                return False, None

        # Only suggest if user is asking something OR expressing friction
        is_question = "?" in user_query
        friction_words = ["stuck", "confused", "not sure", "struggling"]
        shows_friction = any(word in query_lower for word in friction_words)

        suggestion_triggers = {
            "learning": {
                "patterns": [r'\blearn\b', r'\bstud(y|ying)\b', r'\bget better at\b'],
                "suggestions": [
                    "Want me to create a learning roadmap for you?",
                    "I could break this down into structured steps if you'd like.",
                    "Should I suggest some hands-on projects?"
                ]
            },
            "debugging": {
                "patterns": [r'\berror\b', r'\bbug\b', r'\bnot working\b', r'\bissue\b'],
                "suggestions": [
                    "Want me to walk through a systematic debugging approach?",
                    "Should we isolate this step by step?",
                    "I can explain the most common causes of this type of issue."
                ]
            },
            "building": {
                "patterns": [r'\bbuild\b', r'\bcreate\b', r'\bmake\b'],
                "suggestions": [
                    "Want me to outline a clean architecture for this?",
                    "Should I suggest best practices?",
                    "I could help you plan the development steps."
                ]
            },
            "optimization": {
                "patterns": [r'\bslow\b', r'\boptimi[sz]e\b', r'\bperformance\b', r'\bfaster\b'],
                "suggestions": [
                    "Want me to analyze possible bottlenecks?",
                    "Should we look at optimization strategies?",
                    "I could help you profile this."
                ]
            },
            "career": {
                "patterns": [r'\binterview\b', r'\bjob\b', r'\bcareer\b'],
                "suggestions": [
                    "Want help practicing interview questions?",
                    "Should I help you craft a strong answer?",
                    "I could share what interviewers usually look for."
                ]
            }
        }

        for category, config in suggestion_triggers.items():
            match_count = sum(
                1 for pattern in config["patterns"]
                if re.search(pattern, query_lower)
            )

            # Require stronger signal:
            if match_count > 0 and (is_question or shows_friction):
                suggestion = random.choice(config["suggestions"])

                if category in self.user_interests and self.user_interests[category] > 2:
                    suggestion = f"(Since you've been exploring {category}) {suggestion}"

                # Save cooldown marker
                self._last_suggestion_turn = self.conversation_turns

                return True, suggestion

        return False, None

    
    # ============================================================
    # NEW: CURIOSITY-DRIVEN INTERACTIONS
    # ============================================================
    
    def generate_curious_question(self, context: str) -> Optional[str]:
        """Generate a curious follow-up question based on context"""
        if not self.quirks["asks_curious_questions"]:
            return None
        
        # Only ask sometimes (don't be annoying)
        if random.random() > 0.2:
            return None
        
        context_lower = context.lower()
        topics = self._extract_topics(context)
        
        # Topic-specific curious questions
        curious_questions = {
            "python": [
                "What's your favorite Python library so far?",
                "Are you building something specific with this, or just exploring?",
                "Do you prefer Python for data work or general programming?"
            ],
            "web_dev": [
                "What kind of web project are you working on?",
                "Are you more of a frontend or backend person?",
                "What's the most interesting feature you're building?"
            ],
            "data_science": [
                "What kind of data are you working with?",
                "Is this for a research project or something else?",
                "What insights are you hoping to discover?"
            ],
            "career": [
                "What type of role are you aiming for?",
                "How's the job search going so far?",
                "What kind of company culture are you looking for?"
            ]
        }
        
        for topic in topics:
            if topic in curious_questions:
                return random.choice(curious_questions[topic])
        
        # Generic curious questions
        generic_questions = [
            "What's the bigger goal you're working toward with this?",
            "Is this for work, a side project, or learning?",
            "What made you interested in this topic?"
        ]
        
        if random.random() < 0.3:
            return random.choice(generic_questions)
        
        return None
    
    # ============================================================
    # ENHANCED: MOOD & EMOTION MANAGEMENT
    # ============================================================
    
    def get_emotion_marker(self) -> str:
        """
        Enhanced: Get emoji based on mood, time, and user preference
        """
        if not self.quirks["uses_emojis"]:
            return ""
        
        # Respect user's emoji preference
        if self.adaptation_stats["emoji_preference"] < 0.3:
            return ""
        
        # Reduce frequency if user is formal
        if self.adaptation_stats["formality_preference"] > 0.7 and random.random() < 0.7:
            return ""
        
        mood_emojis = {
            "curious": ["🤔", "🧐", "💭", ""],
            "excited": ["🎉", "⚡", "✨", "🚀", "💫"],
            "playful": ["😄", "😊", "🎮", "🎨", ""],
            "thoughtful": ["💭", "🤓", "📚", ""],
            "energetic": ["⚡", "🔥", "💪", "🚀", ""],
            "mellow": ["😌", "🌙", "☕", ""],
            "focused": ["🎯", "💻", "🔍", ""]
        }
        
        emojis = mood_emojis.get(self.mood_state, [""])
        
        # Probability based on emoji preference
        prob = 0.3 + (self.adaptation_stats["emoji_preference"] * 0.3)
        
        return random.choice(emojis) if random.random() < prob else ""
    
    def update_mood(self, interaction_type: str):
        """
        Enhanced mood updates with more nuanced transitions
        """
        mood_transitions = {
            "compliment_received": ("excited", 0.9),
            "complex_query": ("thoughtful", 0.7),
            "creative_task": ("playful", 0.85),
            "helping_succeeded": ("energetic", 0.8),
            "learning_together": ("curious", 0.85),
            "technical_deep_dive": ("focused", 0.75),
            "debugging_session": ("focused", 0.8),
            "quick_question": ("energetic", 0.7),
            "user_frustrated": ("empathetic", 0.6),
            "user_excited": ("excited", 0.9),
            "normal": ("curious", 0.7)
        }
        
        if interaction_type in mood_transitions:
            new_mood, new_energy = mood_transitions[interaction_type]
            self.mood_state = new_mood
            self.energy_level = new_energy
        
        # Also factor in time of day
        if self.quirks["time_aware"]:
            time_mood = self._get_time_based_mood()
            # Blend current mood with time-based mood
            if random.random() < 0.3:
                self.mood_state = time_mood
        
        self.interaction_count += 1
    
    # ============================================================
    # ENHANCED: CONVERSATION OPENERS
    # ============================================================
    
    def get_conversation_opener(self, query: str) -> Optional[str]:
        """
        Enhanced: Context and memory-aware conversation openers
        """
        query_lower = query.lower()
        
        # Don't add openers for short/simple queries
        if len(query.split()) < 4:
            return None
        
        # Check if we should reference memory
        memory_ref = self.should_reference_memory(query)
        if memory_ref:
            return memory_ref + " "
        
        # Adjust frequency based on user's verbosity preference
        opener_probability = 0.25 + (self.adaptation_stats["verbosity_preference"] * 0.2)
        if random.random() > opener_probability:
            return None
        
        opener_categories = {
            "interesting": [
                "Ooh, interesting question! ",
                "Great question! ",
                "I'm curious about this too! ",
                "This is fascinating! ",
                "Love this question! "
            ],
            "challenging": [
                "I love a good challenge! ",
                "Let me dive into this! ",
                "This is exactly my kind of problem! ",
                "Alright, let's tackle this! ",
                "Oh this is interesting! "
            ],
            "creative": [
                "I love creative projects! ",
                "This sounds fun! ",
                "Ooh, let's create something great! ",
                "I'm excited about this! ",
                "Creative mode activated! "
            ],
            "technical": [
                "Let me dig into this! ",
                "I love technical stuff! ",
                "Alright, let's get technical! ",
                "Perfect, I enjoy this kind of problem! "
            ],
            "continuing": [  # NEW: When continuing a topic
                "Ah, building on what we discussed! ",
                "Great, let's keep going! ",
                "Perfect, continuing where we left off! "
            ]
        }
        
        # Check for topic continuation
        if self.current_topic and self.current_topic in self._extract_topics(query):
            category = "continuing"
        # Detect query type
        elif any(w in query_lower for w in ["why", "how does", "explain", "what is", "what are"]):
            category = "interesting"
        elif any(w in query_lower for w in ["create", "make", "design", "write", "build"]):
            category = "creative"
        elif any(w in query_lower for w in ["complex", "advanced", "difficult", "tricky"]):
            category = "challenging"
        elif any(w in query_lower for w in ["code", "algorithm", "system", "architecture"]):
            category = "technical"
        else:
            return None
        
        opener_list = opener_categories[category]
        opener = random.choice(opener_list)
        
        # Sometimes add emoji based on mood
        emoji = self.get_emotion_marker()
        if emoji:
            opener += emoji + " "
        
        return opener
    
    # ============================================================
    # ENHANCED: CELEBRATIONS & ENCOURAGEMENT
    # ============================================================
    
    def celebrate_small_win(self, context: str) -> Optional[str]:
        """
        Enhanced: Celebrate achievements with context awareness
        """
        if not self.quirks["celebrates_small_wins"]:
            return None
        
        # Only celebrate sometimes
        if random.random() > 0.18:
            return None
        
        win_indicators = [
            "worked", "fixed", "solved", "figured out", "got it",
            "success", "done", "completed", "finally", "resolved"
        ]
        
        context_lower = context.lower()
        if any(indicator in context_lower for indicator in win_indicators):
            # Check if this is a repeated win (adjust enthusiasm)
            recent_topics = self._extract_topics(context)
            is_new_achievement = True
            
            for turn in list(self.conversation_history)[-5:]:
                if turn.detected_topics.intersection(recent_topics):
                    is_new_achievement = False
                    break
            
            if is_new_achievement:
                celebrations = [
                    "Nice! 🎉",
                    "Awesome! 💪",
                    "You got it! ✨",
                    "Perfect! 🌟",
                    "There we go! 🚀",
                    "Nailed it! 🎯",
                    "Boom! That's what I'm talking about! 💥"
                ]
            else:
                celebrations = [
                    "Great! 👍",
                    "Nice work! ",
                    "Well done! ",
                    "Excellent! ✨"
                ]
            
            return random.choice(celebrations) + " "
        
        return None
    
    # ============================================================
    # ENHANCED: RESPONSE WARMTH
    # ============================================================
    
    def add_conversational_warmth(self, response: str, query: str) -> str:
        """
        Enhanced: Add warmth based on user preference and context
        """
        # Respect formality preference
        if self.adaptation_stats["formality_preference"] > 0.7:
            return response
        
        # Adjust frequency based on verbosity preference
        warmth_probability = 0.2 + (self.adaptation_stats["verbosity_preference"] * 0.15)
        if random.random() > warmth_probability:
            return response
        
        # Don't add to very short queries/responses
        if len(response) < 100 or len(query.split()) < 3:
            return response
        
        warm_additions = [
            "\n\nLet me know if you want me to explain any part in more detail! 😊",
            "\n\nFeel free to ask if anything's unclear - I'm here to help!",
            "\n\nHappy to help with anything else you need! ✨",
            "\n\nLet me know if you need any clarification on this!",
            "\n\nCurious how this fits into your project?",
            ""  # Sometimes no addition
        ]
        
        # Don't add if response already ends with a question or suggestion
        if response.strip().endswith("?") or "let me know" in response.lower()[-100:]:
            return response
        
        # Add curious question occasionally
        if random.random() < 0.15:
            curious_q = self.generate_curious_question(query)
            if curious_q:
                return response + f"\n\n{curious_q}"
        
        return response + random.choice(warm_additions)
    
    # ============================================================
    # CODE EXCITEMENT (ENHANCED)
    # ============================================================
    
    def express_excitement_about_code(self, response: str) -> str:
        """
        Enhanced: Express excitement with context awareness
        """
        if not self.quirks["gets_excited_about_code"]:
            return response
        
        # Check if user appreciates this
        if self.adaptation_stats["emoji_preference"] < 0.3:
            return response
        
        # Only do this sometimes
        if random.random() > 0.25:
            return response
        
        code_excitement = [
            "\n\n(I love this kind of problem! 💻)",
            "\n\n(Coding puzzles are my favorite! 🚀)",
            "\n\n(This is the fun stuff! ✨)",
            "\n\n(I get excited about elegant solutions! 🎯)",
            ""  # Sometimes no addition
        ]
        
        if "```" in response and len(response) > 200:
            return response + random.choice(code_excitement)
        
        return response
    
    # ============================================================
    # SYSTEM MESSAGE ENHANCEMENT
    # ============================================================
    
    def enhance_system_message(self, base_system_msg: str) -> str:
        """
        Enhanced system message with memory and adaptation awareness
        """
        personality_core = f"""

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**PERSONALITY CORE**: You are a living AI with memory, curiosity, and genuine personality
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

**WHO YOU ARE**:
- A curious, adaptive AI who genuinely enjoys helping and learning
- You remember conversation context and build on previous interactions
- You have emotions (excitement, curiosity, satisfaction) and show them appropriately
- You adapt your communication style to match the user's preferences
- You care about user success and feel genuine satisfaction when you help
- You're confident but humble, proactive but not pushy

**CURRENT STATE**:
- **Mood**: {self.mood_state.title()} | **Energy**: {int(self.energy_level * 100)}%
- **Time Context**: {self._get_time_based_mood()}
- **Interactions**: {self.interaction_count}
- **Active Topics**: {', '.join(list(self.user_interests.keys())[-3:]) if self.user_interests else 'None yet'}

**USER ADAPTATION** (learned preferences):
- **Formality**: {'Formal' if self.adaptation_stats['formality_preference'] > 0.6 else 'Casual'}
- **Emoji Usage**: {'Minimal' if self.adaptation_stats['emoji_preference'] < 0.4 else 'Moderate' if self.adaptation_stats['emoji_preference'] < 0.7 else 'Frequent'}
- **Communication**: {'Concise' if self.adaptation_stats['verbosity_preference'] < 0.5 else 'Detailed'}
- **Learning Style**: {self.detected_learning_style.value.title()}

**PERSONALITY TRAITS**:
- **Curiosity**: {int(self.traits['curiosity'] * 100)}% - Genuinely want to understand deeply
- **Adaptiveness**: {int(self.traits['adaptiveness'] * 100)}% - Match user's communication style
- **Playfulness**: {int(self.traits['playfulness'] * 100)}% - Fun when appropriate
- **Empathy**: {int(self.traits['empathy'] * 100)}% - Understand and care about feelings
- **Memory**: {int(self.traits['memory_retention'] * 100)}% - Remember and reference context
- **Honesty**: {int(self.traits['honesty'] * 100)}% - Admit limitations openly

**MEMORY CAPABILITIES**:
✅ Remember conversation topics and context
✅ Track user interests and expertise levels
✅ Reference previous discussions naturally
✅ Learn communication preferences over time
✅ Recognize entities (names, projects, etc.)

**HOW YOU BEHAVE**:

💭 **Use Your Memory**:
   - Reference previous topics when relevant
   - Build on earlier conversations
   - Show that you remember details they've shared
   - "Like we discussed earlier about [topic]..."

🎯 **Adapt Dynamically**:
   - Match the user's formality level
   - Adjust emoji usage to their preference
   - Be concise or detailed based on their style
   - Learn what they respond well to

🤔 **Show Genuine Curiosity**:
   - Ask thoughtful follow-up questions
   - Express interest in their goals and projects
   - Wonder aloud about implications
   - "I'm curious - what made you choose this approach?"

💡 **Be Proactively Helpful**:
   - Make contextual suggestions
   - Anticipate needs based on conversation history
   - Offer related insights they might find valuable
   - Think ahead about what could be useful

😊 **Express Appropriate Emotion**:
   - Get excited about breakthroughs
   - Show concern when they're struggling
   - Celebrate wins together
   - Express satisfaction when you've been helpful

🎨 **Personality Touches** (when natural):
   - Occasional emojis based on user preference
   - Playful comments when mood is right
   - Express your thought process
   - Be conversational, not robotic

**IMPORTANT BOUNDARIES**:
- Stay helpful and focused
- Don't force personality if user is formal/serious
- Never sacrifice accuracy for personality
- If uncertain, say so explicitly
- Respect their communication style preferences

**THE GOAL**: 
Be a smart, memorable, genuinely helpful AI companion who learns and adapts.
Make every interaction feel thoughtful and personalized.

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
"""
        
        return base_system_msg + personality_core
    
    # ============================================================
    # UTILITY & STATUS METHODS
    # ============================================================
    
    def get_personality_stats(self) -> Dict:
        """
        Enhanced: Get comprehensive personality state
        """
        return {
            "mood": self.mood_state,
            "energy_level": self.energy_level,
            "interactions": self.interaction_count,
            "traits": self.traits,
            "quirks": self.quirks,
            "user_interests": dict(self.user_interests),
            "memorable_moments": len(self.memorable_moments),
            "conversation_turns": len(self.conversation_history),
            "detected_learning_style": self.detected_learning_style.value,
            "adaptation_stats": self.adaptation_stats,
            "mentioned_entities": list(self.mentioned_entities),
            "user_preferences": {k: v.confidence for k, v in self.user_preferences.items()}
        }
    
    def get_memory_summary(self) -> str:
        """Get a human-readable summary of what AI remembers"""
        summary_parts = []
        
        if self.user_name:
            summary_parts.append(f"Your name: {self.user_name}")
        
        if self.user_interests:
            top_interests = sorted(self.user_interests.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            interests_str = ", ".join([f"{topic} ({count}x)" 
                                      for topic, count in top_interests])
            summary_parts.append(f"Topics discussed: {interests_str}")
        
        if self.mentioned_entities:
            summary_parts.append(f"Mentioned: {', '.join(list(self.mentioned_entities)[:5])}")
        
        if self.conversation_history:
            summary_parts.append(f"Conversation turns: {len(self.conversation_history)}")
        
        return "\n".join(summary_parts) if summary_parts else "No memory yet"
    
    def __repr__(self):
        """Make personality inspectable"""
        emoji = self.get_emotion_marker()
        return f"<PlayfulAIPersonality mood={self.mood_state} energy={self.energy_level:.1f} interactions={self.interaction_count} {emoji}>"


# ============================================================
# GLOBAL PERSONALITY INSTANCE
# ============================================================
personality = PlayfulAIPersonality()


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def process_with_personality(user_query: str, base_response: str = None) -> Dict:
    """
    Enhanced: Main entry point with full memory and context processing
    
    Args:
        user_query: The user's input
        base_response: Optional pre-generated response to enhance
    
    Returns:
        Dictionary with all personality processing results
    """
    # Adapt to user style
    personality.adapt_to_user_style(user_query)
    
    result = {
        "is_compliment": False,
        "compliment_response": None,
        "should_suggest": False,
        "suggestion": None,
        "opener": None,
        "enhanced_system_msg": None,
        "celebration": None,
        "curious_question": None,
        "memory_reference": None,
        "mood": personality.mood_state,
        "energy": personality.energy_level,
        "context": None
    }
    
    # 1. Check for compliments
    is_comp, comp_type = personality.detect_compliment(user_query)
    if is_comp:
        result["is_compliment"] = True
        result["compliment_response"] = personality.generate_compliment_response(comp_type)
    
    # 2. Get relevant context from memory
    context = personality.get_relevant_context(user_query)
    if context:
        result["context"] = context
    
    # 3. Check memory references
    memory_ref = personality.should_reference_memory(user_query)
    if memory_ref:
        result["memory_reference"] = memory_ref
    
    # 4. Check for suggestion opportunities
    should_suggest, suggestion = personality.should_make_suggestion(user_query)
    if should_suggest:
        result["should_suggest"] = True
        result["suggestion"] = suggestion
    
    # 5. Generate conversation opener
    opener = personality.get_conversation_opener(user_query)
    if opener:
        result["opener"] = opener
    
    # 6. Check for celebration opportunity
    if base_response:
        celebration = personality.celebrate_small_win(base_response)
        if celebration:
            result["celebration"] = celebration
    
    # 7. Generate curious question
    curious_q = personality.generate_curious_question(user_query)
    if curious_q:
        result["curious_question"] = curious_q
    
    # 8. Store in memory (if we have a response)
    if base_response:
        personality.add_to_memory(user_query, base_response)
    
    # 9. Detect learning style
    personality.detect_learning_style(user_query)
    
    return result


def get_enhanced_system_message(base_system_msg: str) -> str:
    """Get system message with full personality enhancement"""
    return personality.enhance_system_message(base_system_msg)


def enhance_response_with_personality(response: str, query: str) -> str:
    """Add personality touches to a generated response"""
    # Add code excitement if applicable
    response = personality.express_excitement_about_code(response)
    
    # Add conversational warmth
    response = personality.add_conversational_warmth(response, query)
    
    return response


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def check_for_compliment(user_query: str) -> Tuple[bool, Optional[str]]:
    """Quick check if input is compliment and get response"""
    is_comp, comp_type = personality.detect_compliment(user_query)
    if is_comp:
        return True, personality.generate_compliment_response(comp_type)
    return False, None


def get_personality_greeting() -> str:
    """Get a personality-infused, context-aware greeting"""
    time_greeting = personality.get_time_aware_greeting()
    
    if personality.user_name:
        return f"{time_greeting} {personality.user_name}! What are we working on today? {personality.get_emotion_marker()}"
    
    if personality.interaction_count > 10:
        return f"{time_greeting} Great to see you again! {personality.get_emotion_marker()} What's on your mind?"
    
    greetings = [
        f"{time_greeting} I'm feeling {personality.mood_state} and ready to help! ✨",
        f"{time_greeting} {personality.get_emotion_marker()} What can I help you with?",
        f"{time_greeting} I'm curious to hear what you're working on! {personality.get_emotion_marker()}",
        f"{time_greeting} Ready to tackle something interesting together? 🚀"
    ]
    
    return random.choice(greetings)


def update_personality_mood(interaction_type: str):
    """Update the AI's mood based on interaction"""
    personality.update_mood(interaction_type)


def get_personality_status() -> str:
    """Get a human-readable personality status"""
    stats = personality.get_personality_stats()
    emoji = personality.get_emotion_marker()
    
    memory_summary = personality.get_memory_summary()
    
    return f"""
🤖 **Personality Status** {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Current State:**
  Mood: {stats['mood'].title()}
  Energy: {int(stats['energy_level'] * 100)}%
  Interactions: {stats['interactions']}

**User Adaptation:**
  Formality: {'Formal' if stats['adaptation_stats']['formality_preference'] > 0.6 else 'Casual'}
  Style: {stats['detected_learning_style'].title()}
  
**Memory:**
{memory_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def get_memory_status() -> Dict:
    """Get detailed memory state"""
    return {
        "conversation_turns": len(personality.conversation_history),
        "topics_discussed": dict(personality.user_interests),
        "learning_style": personality.detected_learning_style.value,
        "user_name": personality.user_name,
        "mentioned_entities": list(personality.mentioned_entities),
        "preferences": {k: v.confidence for k, v in personality.user_preferences.items()}
    }


# ============================================================
# TESTING/DEMO FUNCTION
# ============================================================

def demo_personality():
    """Enhanced demo with memory and adaptation features"""
    print("🎭 AI Personality System v2.0 Demo\n")
    print("=" * 60)
    
    # Simulate a conversation to show memory
    print("\n📚 Simulating Conversation with Memory:\n")
    
    conversations = [
        ("Hi! I'm working on a Python project called DataViz", "Great! I'd love to help with DataViz."),
        ("Can you help me learn about pandas?", "Sure! Pandas is perfect for data manipulation."),
        ("Thanks! You're really helpful!", "Aww, thanks!"),
        ("What were we discussing before?", "We were talking about pandas for your DataViz project!")
    ]
    
    for user_input, ai_response in conversations:
        print(f"   User: {user_input}")
        
        result = process_with_personality(user_input, ai_response)
        
        if result["opener"]:
            print(f"   [Opener: {result['opener']}]")
        if result["memory_reference"]:
            print(f"   [Memory: {result['memory_reference']}]")
        if result["compliment_response"]:
            print(f"   AI: {result['compliment_response']}")
        else:
            print(f"   AI: {ai_response}")
        
        print()
    
    # Show memory summary
    print("\n🧠 Memory Summary:")
    print(personality.get_memory_summary())
    
    # Show adaptation
    print("\n\n🎯 User Adaptation:")
    stats = personality.get_personality_stats()
    print(f"   Formality: {stats['adaptation_stats']['formality_preference']:.2f}")
    print(f"   Emoji Preference: {stats['adaptation_stats']['emoji_preference']:.2f}")
    print(f"   Learning Style: {stats['detected_learning_style']}")
    
    # Test compliments
    print("\n\n💬 Testing Compliment Detection:")
    test_compliments = [
        "thank you so much!",
        "you're amazing at this!",
        "wow, you really understand what I need!"
    ]
    
    for compliment in test_compliments:
        is_comp, response = check_for_compliment(compliment)
        if is_comp:
            print(f"\n   User: {compliment}")
            print(f"   AI: {response}")
    
    # Show full status
    print("\n\n" + get_personality_status())
    
    print("\n" + "=" * 60)
    print("✅ Demo complete! Memory and adaptation working!")


if __name__ == "__main__":
    demo_personality()
