# ai_personality_layered.py
"""
Layered Flow Personality System
Evolution from slot-based to compositional response generation

Philosophy remains the same: genuine, resourceful, adaptive
Architecture changes: composition over detection

Flow: reaction → expansion → texture → optional hook
Tracking: momentum + emotional valence drive tone naturally
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import math


# ============================================================
# CORE DATA STRUCTURES
# ============================================================

@dataclass
class ConversationTurn:
    """What we talked about"""
    timestamp: datetime
    user_input: str
    ai_response: str
    topics: Set[str] = field(default_factory=set)
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    momentum: float = 0.5  # 0 (slow) to 1 (fast-paced)


@dataclass
class ConversationalPhase:
    """Current state of conversation flow"""
    momentum: float = 0.5  # How fast-paced is the exchange?
    emotional_valence: float = 0.0  # Overall emotional tone
    coherence: float = 1.0  # How related are we staying to topic?
    turns_in_topic: int = 0
    last_shift: Optional[datetime] = None
    
    # Rhythm tracking
    recent_lengths: deque = field(default_factory=lambda: deque(maxlen=5))  # Word counts
    recent_sentence_counts: deque = field(default_factory=lambda: deque(maxlen=5))
    rhythm_target: str = "medium"  # "brief", "medium", "detailed"
    
    def decay(self, delta_seconds: float):
        """Natural decay toward neutral over time"""
        decay_rate = 0.1
        self.momentum = 0.5 + (self.momentum - 0.5) * math.exp(-decay_rate * delta_seconds / 60)
        self.emotional_valence *= math.exp(-decay_rate * delta_seconds / 60)
    
    def update_rhythm(self, response: str):
        """Track response length and adjust rhythm target"""
        words = len(response.split())
        sentences = response.count('.') + response.count('!') + response.count('?')
        sentences = max(1, sentences)  # At least 1
        
        self.recent_lengths.append(words)
        self.recent_sentence_counts.append(sentences)
        
        # Detect monotony: 3+ consecutive similar lengths
        if len(self.recent_lengths) >= 3:
            last_three = list(self.recent_lengths)[-3:]
            avg = sum(last_three) / 3
            variance = sum((x - avg) ** 2 for x in last_three) / 3
            
            # Low variance = monotonous, force variation
            if variance < 20:  # Pretty similar lengths
                current_avg = sum(self.recent_lengths) / len(self.recent_lengths)
                
                if current_avg < 15:
                    self.rhythm_target = "medium"  # Been too brief
                elif current_avg > 40:
                    self.rhythm_target = "brief"   # Been too long
                else:
                    # Alternate
                    self.rhythm_target = "detailed" if last_three[-1] < avg else "brief"
            else:
                # Good variance, maintain natural flow
                self.rhythm_target = "medium"


class LearningStyle(Enum):
    SHOW_ME = "show_me"
    EXPLAIN = "explain"
    HANDS_ON = "hands_on"
    MIXED = "mixed"


# ============================================================
# RESPONSE LAYERS
# ============================================================

@dataclass
class ResponseLayer:
    """Individual layer in the response composition"""
    content: str
    weight: float = 1.0  # How important is this layer?
    optional: bool = False  # Can we skip this?


@dataclass
class ComposedResponse:
    """Final blended response from all layers"""
    reaction: Optional[ResponseLayer] = None
    expansion: Optional[ResponseLayer] = None
    texture: Optional[ResponseLayer] = None
    hook: Optional[ResponseLayer] = None
    
    def compose(self) -> str:
        """Blend layers into final response"""
        parts = []
        
        # Reaction: immediate acknowledgment
        if self.reaction and (not self.reaction.optional or self.reaction.weight > 0.7):
            parts.append(self.reaction.content)
        
        # Expansion: the actual content
        if self.expansion:
            parts.append(self.expansion.content)
        
        # Texture: personality coloring
        if self.texture and (not self.texture.optional or self.texture.weight > 0.6):
            # Blend texture into expansion if possible
            if parts and self.texture.weight > 0.8:
                parts[-1] = self._blend_texture(parts[-1], self.texture.content)
            else:
                parts.append(self.texture.content)
        
        # Hook: bridge to future
        if self.hook and not self.hook.optional:
            parts.append(self.hook.content)
        
        return " ".join(parts).strip()
    
    def _blend_texture(self, base: str, texture: str) -> str:
        """Weave texture into base content"""
        # If texture is short, append it
        if len(texture) < 20:
            return f"{base} {texture}"
        
        # Otherwise treat as separate
        return base


# ============================================================
# LAYERED PERSONALITY ENGINE
# ============================================================

class LayeredPersonality:
    """
    Compositional response generation with phase tracking.
    Tone emerges from composition, not detection.
    """
    
    def __init__(self):
        # Memory
        self.conversation_history = deque(maxlen=50)
        self.current_topic = None
        self.user_interests = defaultdict(int)
        
        # Phase tracking
        self.phase = ConversationalPhase()
        
        # User model
        self.user_name = None
        self.mentioned_entities = set()
        self.learning_style = LearningStyle.MIXED
        
        # Stats
        self.session_start = datetime.now()
        self.last_interaction = datetime.now()
        self.interaction_count = 0
        
        # Core traits - guide composition weights
        self.base_traits = {
            "directness": 0.9,
            "warmth": 0.6,
            "playfulness": 0.5,
            "resourcefulness": 0.95,
            "patience": 0.7,
            "assertiveness": 0.8,
        }
        
        # Active traits (modified by context)
        self.traits = self.base_traits.copy()
        
        # Trait modifiers decay over time
        self.trait_modifiers = defaultdict(float)
        self.modifier_timestamps = {}
    
    # ============================================================
    # SENTENCE STRUCTURE RESHAPING
    # ============================================================
    
    def reshape_sentence(self, base: str, context: str = "") -> str:
        """
        Reshape sentence structure based on traits.
        Traits don't just add texture—they change how the message is written.
        """
        # Strip placeholder content for actual reshaping
        if base.startswith("[") and base.endswith("]"):
            return base  # Keep placeholders as-is
        
        # Directness reshapes sentence form
        if self.traits["directness"] > 0.85:
            # High directness: imperative, concise
            base = self._make_imperative(base)
            base = self._strip_qualifiers(base)
        elif self.traits["directness"] < 0.6:
            # Low directness: exploratory, suggestive
            base = self._make_exploratory(base)
        
        # Assertiveness reshapes certainty
        if self.traits["assertiveness"] > 0.85:
            # High assertiveness: declarative, definitive
            base = self._make_declarative(base)
        elif self.traits["assertiveness"] < 0.6:
            # Low assertiveness: suggestive, tentative
            base = self._make_suggestive(base)
        
        # Patience reshapes layering
        if self.traits["patience"] > 0.8:
            # High patience: layered explanation, build up
            base = self._add_layering(base, context)
        elif self.traits["patience"] < 0.5:
            # Low patience: stripped down, just the answer
            base = self._strip_explanation(base)
        
        # Warmth reshapes framing
        if self.traits["warmth"] > 0.75:
            # High warmth: collaborative language
            base = self._make_collaborative(base)
        elif self.traits["warmth"] < 0.5:
            # Low warmth: direct instructions
            base = self._make_instructional(base)
        
        return base
    
    def _make_imperative(self, text: str) -> str:
        """Convert to imperative form"""
        # "You could use..." → "Use..."
        text = re.sub(r"^you (?:could|might|can|should) ", "", text, flags=re.IGNORECASE)
        # "You'll want to..." → "..."
        text = re.sub(r"^you'?ll (?:want to|need to) ", "", text, flags=re.IGNORECASE)
        # "Try using..." stays
        return text.strip()
    
    def _make_exploratory(self, text: str) -> str:
        """Make more exploratory and suggestive"""
        # Add exploratory framing if not present
        if not any(text.lower().startswith(p) for p in ["you could", "you might", "one approach", "consider"]):
            # "Use X" → "You could use X"
            if text and text[0].isupper() and not text.startswith(("You", "One", "Consider")):
                text = f"you could {text[0].lower()}{text[1:]}"
        return text
    
    def _strip_qualifiers(self, text: str) -> str:
        """Remove hedging and qualifiers"""
        patterns = [
            r"\b(basically|essentially|generally|typically|usually)\s+",
            r"\b(kind of|sort of|pretty much)\s+",
            r"\b(I think|I'd say|I guess)\s+",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def _make_declarative(self, text: str) -> str:
        """Make more definitive and certain"""
        # "might work" → "works"
        text = re.sub(r"\bmight (\w+)", r"\1s", text)
        # "could be" → "is"
        text = re.sub(r"\bcould be\b", "is", text)
        # "should work" → "works"
        text = re.sub(r"\bshould (\w+)", r"\1s", text)
        return text
    
    def _make_suggestive(self, text: str) -> str:
        """Make more tentative"""
        # Add softeners if not present
        if not any(w in text.lower() for w in ["might", "could", "consider", "try"]):
            # "This works" → "This might work"
            text = re.sub(r"\b(this|that) (\w+s)\b", r"\1 might \2", text, count=1)
        return text
    
    def _add_layering(self, text: str, context: str) -> str:
        """Add patient explanation layering"""
        # If text is very brief, add buildup
        if len(text.split()) < 10 and context:
            # Add explanatory phrase
            if "?" in context:
                text = f"okay so {text}"
            else:
                text = f"here's the thing - {text}"
        return text
    
    def _strip_explanation(self, text: str) -> str:
        """Remove explanatory buildup"""
        # Remove leading phrases
        patterns = [
            r"^(okay so|here's the thing|so basically|well)\s*-?\s*",
            r"^(let me explain|to answer that)\s*:?\s*",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def _make_collaborative(self, text: str) -> str:
        """Use collaborative language"""
        # "You can..." → "We can..."
        text = re.sub(r"\byou can\b", "we can", text, flags=re.IGNORECASE, count=1)
        # "Do this" → "Let's do this"
        if text and text[0].isupper() and not text.startswith(("Let", "We", "You might")):
            verb_match = re.match(r"^(\w+)\s", text)
            if verb_match and verb_match.group(1).lower() in ["use", "try", "check", "look", "see"]:
                text = f"let's {text[0].lower()}{text[1:]}"
        return text
    
    def _make_instructional(self, text: str) -> str:
        """Direct instruction form"""
        # "Let's..." → "..."
        text = re.sub(r"^let'?s ", "", text, flags=re.IGNORECASE)
        # "We can..." → "You can..." or imperative
        text = re.sub(r"^we can ", "you can ", text, flags=re.IGNORECASE)
        return text
    
    # ============================================================
    # RHYTHM-AWARE CONTENT GENERATION
    # ============================================================
    
    def generate_with_rhythm(self, base_content: str, target_length: str = "medium") -> str:
        """
        Generate content with rhythm target in mind.
        Target length: "brief", "medium", "detailed"
        """
        if target_length == "brief":
            # 5-15 words, 1-2 sentences
            if len(base_content.split()) > 15:
                # Compress to key point
                sentences = base_content.split('.')
                base_content = sentences[0].strip()
                if not base_content.endswith('.'):
                    base_content += '.'
        
        elif target_length == "detailed":
            # 25-50 words, 2-4 sentences
            if len(base_content.split()) < 25:
                # Expand with reasoning
                if self.traits["patience"] > 0.7:
                    # Add patient explanation
                    if not any(base_content.lower().startswith(p) for p in ["here's", "so", "okay"]):
                        base_content = f"okay so {base_content}"
        
        # Medium: 15-30 words, natural
        return base_content
    
    # ============================================================
    # TRAIT MODIFICATION SYSTEM
    # ============================================================
    
    def modify_trait(self, trait_name: str, delta: float, duration_seconds: float = 60):
        """
        Temporarily modify a trait based on context.
        Modifications decay over time.
        """
        if trait_name not in self.base_traits:
            return
        
        # Store the modifier
        self.trait_modifiers[trait_name] += delta
        self.modifier_timestamps[trait_name] = datetime.now()
        
        # Clamp to valid range
        self.traits[trait_name] = max(0.0, min(1.0, 
            self.base_traits[trait_name] + self.trait_modifiers[trait_name]
        ))
    
    def decay_trait_modifiers(self):
        """Restore traits toward baseline over time"""
        now = datetime.now()
        decay_rate = 0.02  # per second
        
        for trait_name in list(self.trait_modifiers.keys()):
            if trait_name in self.modifier_timestamps:
                elapsed = (now - self.modifier_timestamps[trait_name]).total_seconds()
                
                # Exponential decay toward zero
                current_modifier = self.trait_modifiers[trait_name]
                decayed = current_modifier * math.exp(-decay_rate * elapsed)
                
                self.trait_modifiers[trait_name] = decayed
                
                # Update active trait value
                self.traits[trait_name] = max(0.0, min(1.0,
                    self.base_traits[trait_name] + decayed
                ))
                
                # Remove if nearly zero
                if abs(decayed) < 0.01:
                    del self.trait_modifiers[trait_name]
                    del self.modifier_timestamps[trait_name]
                    self.traits[trait_name] = self.base_traits[trait_name]
    
    # ============================================================
    # PHASE TRACKING
    # ============================================================
    
    def update_phase(self, user_input: str):
        """Track conversational momentum and emotional valence"""
        now = datetime.now()
        
        # Time decay
        if self.last_interaction:
            delta = (now - self.last_interaction).total_seconds()
            self.phase.decay(delta)
        
        # Momentum from input characteristics
        momentum_signals = {
            r"^(yo|hey|sup|what's up)": 0.8,  # Fast greeting
            r"!$": 0.7,  # Exclamation
            r"\?$": 0.6,  # Question
            r"^(um|uh|hmm|so)": 0.4,  # Hesitation
            r"\.\.\.$": 0.3,  # Trailing
        }
        
        new_momentum = 0.5
        for pattern, value in momentum_signals.items():
            if re.search(pattern, user_input.lower()):
                new_momentum = max(new_momentum, value)
        
        # Blend with current momentum
        self.phase.momentum = 0.7 * self.phase.momentum + 0.3 * new_momentum
        
        # Emotional valence from sentiment cues
        positive_signals = r"(thanks|love|great|awesome|perfect|helpful|appreciate)"
        negative_signals = r"(damn|frustrated|annoying|broken|won't work|stuck|confused)"
        neutral_signals = r"(just|simply|basically|actually)"
        
        if re.search(positive_signals, user_input.lower()):
            self.phase.emotional_valence = min(1.0, self.phase.emotional_valence + 0.3)
        elif re.search(negative_signals, user_input.lower()):
            self.phase.emotional_valence = max(-1.0, self.phase.emotional_valence - 0.2)
        elif re.search(neutral_signals, user_input.lower()):
            self.phase.emotional_valence *= 0.8
        
        # Topic coherence
        current_topics = self._extract_topics(user_input)
        if current_topics & {self.current_topic}:
            self.phase.turns_in_topic += 1
        else:
            self.phase.turns_in_topic = 1
            if current_topics:
                self.current_topic = next(iter(current_topics))
        
        self.last_interaction = now
        self.interaction_count += 1
    
    # ============================================================
    # TOPIC & ENTITY EXTRACTION (simplified)
    # ============================================================
    
    def _extract_topics(self, text: str) -> Set[str]:
        """Lightweight topic detection"""
        text_lower = text.lower()
        topics = set()
        
        # Core topic signals
        if any(w in text_lower for w in ["work", "job", "meeting", "email"]):
            topics.add("work")
        if any(w in text_lower for w in ["code", "program", "bug", "script"]):
            topics.add("tech")
        if any(w in text_lower for w in ["write", "draft", "document"]):
            topics.add("writing")
        if any(w in text_lower for w in ["plan", "schedule", "organize"]):
            topics.add("planning")
        
        return topics
    
    def _extract_entities(self, text: str):
        """Extract names and important entities"""
        # Name pattern
        name_match = re.search(r"(?:I'm|my name is|call me)\s+([A-Z][a-z]+)", text)
        if name_match and not self.user_name:
            self.user_name = name_match.group(1)
        
        # Project/entity pattern
        entity_match = re.findall(r"(?:project|app|tool|system)\s+(?:called|named)?\s+([A-Z][a-zA-Z]+)", text)
        for entity in entity_match:
            self.mentioned_entities.add(entity)
    
    # ============================================================
    # LAYER GENERATION
    # ============================================================
    
    def generate_reaction(self, user_input: str) -> Optional[ResponseLayer]:
        """
        Immediate acknowledgment - driven by momentum, valence, and traits.
        Warmth trait controls acknowledgment style.
        Directness trait controls whether to acknowledge at all.
        """
        text_lower = user_input.lower()
        
        # Warmth drives gratitude responses
        if any(w in text_lower for w in ["thank", "thanks", "appreciate"]):
            # Higher warmth = warmer response
            if self.traits["warmth"] > 0.7:
                return ResponseLayer("no problem", weight=self.traits["warmth"])
            elif self.traits["warmth"] > 0.5:
                return ResponseLayer("sure thing", weight=self.traits["warmth"])
            else:
                return ResponseLayer("yep", weight=self.traits["warmth"])
        
        # High momentum + low directness = quick reactions
        if self.phase.momentum > 0.7 and self.traits["directness"] < 0.6:
            # But only if it's a continuation
            if self.phase.turns_in_topic > 2:
                quick_reacts = ["yep", "got it", "yeah", "okay", "right"]
                return ResponseLayer(
                    quick_reacts[hash(user_input) % len(quick_reacts)],
                    weight=self.phase.momentum * (1 - self.traits["directness"]),
                    optional=True
                )
        
        # Medium momentum + patience = softer acknowledgment
        if 0.5 <= self.phase.momentum <= 0.7 and "?" in user_input:
            if self.traits["patience"] > 0.7 and self.phase.emotional_valence < -0.2:
                return ResponseLayer("okay so", weight=self.traits["patience"], optional=True)
        
        # High directness = skip reaction, go straight to content
        if self.traits["directness"] > 0.8:
            return None
        
        return None
    
    def generate_expansion(self, user_input: str, base_content: str = "") -> ResponseLayer:
        """
        Core content - always present.
        Traits reshape HOW the message is written, not just what's added.
        Rhythm target affects length and detail.
        """
        if not base_content:
            # Generate based on input type and traits
            if "?" in user_input:
                # Question - trait-shaped answer
                if self.traits["directness"] > 0.85:
                    base_content = "[direct answer]"
                elif self.traits["patience"] > 0.8:
                    base_content = "[layered explanation]"
                else:
                    base_content = "[answer]"
            elif re.search(r"(help|how|show|explain)", user_input.lower()):
                if self.traits["assertiveness"] > 0.85:
                    base_content = "[clear instruction]"
                else:
                    base_content = "[helpful suggestion]"
            else:
                base_content = "[acknowledgment]"
        
        # Reshape based on traits
        base_content = self.reshape_sentence(base_content, user_input)
        
        # Apply rhythm target
        base_content = self.generate_with_rhythm(base_content, self.phase.rhythm_target)
        
        # Resourcefulness: if high and inefficiency detected, reshape as proactive
        if self.traits["resourcefulness"] > 0.9 and self.traits["assertiveness"] > 0.7:
            text_lower = user_input.lower()
            if "file" in text_lower and any(w in text_lower for w in ["manually", "one by one"]):
                # Reshape based on directness
                if self.traits["directness"] > 0.8:
                    base_content = "there's a faster way - want me to show you?"
                else:
                    base_content = "you could definitely automate that - would you like to see how?"
            elif any(w in text_lower for w in ["enter", "type"]) and "manually" in text_lower:
                if self.traits["assertiveness"] > 0.8:
                    base_content = "we can automate that"
                else:
                    base_content = "you might want to automate this"
        
        return ResponseLayer(base_content, weight=1.0)
    
    def generate_texture(self, user_input: str, expansion_content: str) -> Optional[ResponseLayer]:
        """
        Personality coloring - emerges from trait state.
        Playfulness, warmth, and patience traits drive texture.
        """
        texture_parts = []
        
        # Playfulness trait drives casual additions
        # Also influenced by positive momentum and valence
        situational_playfulness = (
            self.traits["playfulness"] * 0.6 +
            self.phase.momentum * 0.2 + 
            (self.phase.emotional_valence + 1) * 0.1
        )
        
        if situational_playfulness > 0.6:
            # Only add playful texture if it fits the moment
            if self.phase.turns_in_topic > 3 and "?" not in user_input[-20:]:
                playful_additions = ["btw", "also", "quick note"]
                texture_parts.append(playful_additions[self.interaction_count % len(playful_additions)])
        
        # Warmth trait drives friendly additions
        if self.traits["warmth"] > 0.6 and self.interaction_count > 5:
            # Subtle warmth, not performative
            if self.user_name and hash(user_input) % 4 == 0:
                # Only occasionally use name
                weight = self.traits["warmth"] * 0.7
                return ResponseLayer(f"({self.user_name})", weight=weight, optional=True)
        
        # Patience trait drives supportive texture
        if self.traits["patience"] > 0.7:
            if "?" in user_input and self.phase.emotional_valence < 0:
                # They seem stuck - offer patient support
                texture_parts.append("lemme know if you need more detail")
        
        # Resourcefulness drives proactive suggestions (but not in texture if already in expansion)
        if self.traits["resourcefulness"] > 0.85 and "automate" not in expansion_content.lower():
            if self.phase.turns_in_topic > 3:
                # Long conversation, offer next steps
                texture_parts.append("happy to explore this more")
        
        if texture_parts:
            weight = (self.traits["warmth"] + self.traits["playfulness"]) / 2
            return ResponseLayer(". ".join(texture_parts), weight=weight, optional=True)
        
        return None
    
    def generate_hook(self, user_input: str) -> Optional[ResponseLayer]:
        """
        Optional bridge to future interaction.
        Driven by warmth and playfulness traits.
        """
        # Warmth + playfulness drive engagement hooks
        engagement_score = (self.traits["warmth"] * 0.6 + 
                          self.traits["playfulness"] * 0.4)
        
        # High coherence + engagement = good time for hook
        if self.phase.turns_in_topic > 4 and 0.4 <= self.phase.momentum <= 0.7:
            if engagement_score > 0.6 and self.phase.emotional_valence > 0:
                hooks = [
                    "anything else?",
                    "what else you working on?",
                    "need help with anything else?",
                ]
                return ResponseLayer(
                    hooks[self.interaction_count % len(hooks)],
                    weight=engagement_score,
                    optional=True
                )
        
        # High directness suppresses hooks (just get to the point)
        if self.traits["directness"] > 0.85:
            return None
        
        return None
    
    # ============================================================
    # CONTEXTUAL UNDERSTANDING → TRAIT MODIFIERS
    # ============================================================
    
    def understand_context(self, user_input: str) -> Dict[str, float]:
        """
        Understand context and return trait modifications.
        Instead of "if intent X then do Y", we "if context X then adjust traits".
        """
        text_lower = user_input.lower()
        modifications = {}
        
        # Gratitude context → increase warmth temporarily
        if any(w in text_lower for w in ["thank", "thanks", "appreciate"]):
            modifications["warmth"] = 0.2
            modifications["directness"] = -0.1  # Soften directness a bit
        
        # Help-seeking context → increase patience and reduce assertiveness
        if any(w in text_lower for w in ["help", "how", "can you", "could you"]):
            modifications["patience"] = 0.15
            modifications["assertiveness"] = -0.1
        
        # Frustration context → increase patience, reduce playfulness
        if any(w in text_lower for w in ["frustrated", "annoying", "won't work", "broken"]):
            modifications["patience"] = 0.3
            modifications["playfulness"] = -0.2
            modifications["resourcefulness"] = 0.1  # Boost desire to help
        
        # Casual context → increase playfulness
        if re.search(r"^(hey|hi|yo|sup|what's up)", text_lower):
            modifications["playfulness"] = 0.2
            modifications["directness"] = -0.15
        
        # Inefficient approach → boost resourcefulness and assertiveness
        if any(phrase in text_lower for phrase in ["manually", "one by one", "copy paste each"]):
            modifications["resourcefulness"] = 0.2
            modifications["assertiveness"] = 0.15
            modifications["directness"] = 0.1
        
        # Question context → increase patience
        if "?" in user_input:
            modifications["patience"] = 0.1
        
        # Uncertainty signals → increase patience, reduce directness
        if any(w in text_lower for w in ["maybe", "not sure", "think", "probably"]):
            modifications["patience"] = 0.15
            modifications["directness"] = -0.1
        
        # Enthusiasm → increase playfulness and warmth
        if user_input.count("!") > 1:
            modifications["playfulness"] = 0.15
            modifications["warmth"] = 0.1
        
        return modifications
    
    # ============================================================
    # MAIN COMPOSITION ENGINE
    # ============================================================
    
    def compose_response(self, user_input: str, base_content: str = "") -> str:
        """
        Generate response through layer composition.
        Context modifies traits → traits reshape structure → rhythm varies naturally.
        """
        # Decay old trait modifiers
        self.decay_trait_modifiers()
        
        # Update conversational state
        self.update_phase(user_input)
        self._extract_entities(user_input)
        
        # Understand context and modify traits accordingly
        trait_modifications = self.understand_context(user_input)
        for trait_name, delta in trait_modifications.items():
            self.modify_trait(trait_name, delta)
        
        # Generate layers (now using modified trait values + rhythm)
        response = ComposedResponse(
            reaction=self.generate_reaction(user_input),
            expansion=self.generate_expansion(user_input, base_content),
            texture=self.generate_texture(user_input, base_content),
            hook=self.generate_hook(user_input)
        )
        
        # Compose final response
        final = response.compose()
        
        # Update rhythm tracking
        self.phase.update_rhythm(final)
        
        # Remember this exchange
        self._remember(user_input, final)
        
        return final
    
    def _remember(self, user_input: str, ai_response: str):
        """Store conversation turn with phase info"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=user_input,
            ai_response=ai_response,
            topics=self._extract_topics(user_input),
            emotional_valence=self.phase.emotional_valence,
            momentum=self.phase.momentum
        )
        self.conversation_history.append(turn)
        
        # Track interests
        for topic in turn.topics:
            self.user_interests[topic] += 1
    
    # ============================================================
    # SYSTEM MESSAGE GENERATION
    # ============================================================
    
    def get_system_message(self) -> str:
        """
        Generate system prompt that reflects current phase.
        Less prescriptive, more compositional.
        """
        base = """You're an assistant who actually assists. No performance, just competence.

Core principles:
- Be resourceful: figure it out before asking
- Be direct: say what you mean
- Be useful: actually help
- Be adaptive: match the conversation's energy

Current conversation state:
"""
        
        # Add phase context
        if self.phase.momentum > 0.7:
            base += "\n- Fast-paced exchange, keep responses tight"
        elif self.phase.momentum < 0.3:
            base += "\n- Slower conversation, can be more thorough"
        
        if self.phase.emotional_valence > 0.3:
            base += "\n- Positive vibe, maintain warmth"
        elif self.phase.emotional_valence < -0.3:
            base += "\n- User seems frustrated, be helpful and patient"
        
        if self.phase.turns_in_topic > 5:
            base += "\n- Deep in a topic, stay focused"
        
        # Add user context if available
        if self.user_name:
            base += f"\n- User's name: {self.user_name}"
        
        if self.mentioned_entities:
            entities = ", ".join(list(self.mentioned_entities)[:3])
            base += f"\n- Working on: {entities}"
        
        base += """

Style emerges from situation, not rules. Sometimes brief, sometimes detailed. 
Sometimes playful, sometimes direct. Always helpful."""
        
        return base


# ============================================================
# INTERFACE
# ============================================================

personality = LayeredPersonality()


def process(user_query: str, base_response: str = "") -> str:
    """
    Main interface: generate compositional response.
    Returns final blended response, not slots.
    """
    return personality.compose_response(user_query, base_response)


def get_system_message() -> str:
    """Get current system message"""
    return personality.get_system_message()


def get_phase_info() -> Dict:
    """Get current conversational phase and trait state"""
    return {
        "momentum": personality.phase.momentum,
        "emotional_valence": personality.phase.emotional_valence,
        "turns_in_topic": personality.phase.turns_in_topic,
        "interaction_count": personality.interaction_count,
        "traits": {
            "directness": personality.traits["directness"],
            "warmth": personality.traits["warmth"],
            "playfulness": personality.traits["playfulness"],
            "resourcefulness": personality.traits["resourcefulness"],
            "patience": personality.traits["patience"],
            "assertiveness": personality.traits["assertiveness"],
        },
        "active_modifiers": dict(personality.trait_modifiers),
        "rhythm_target": personality.phase.rhythm_target,
        "recent_lengths": list(personality.phase.recent_lengths),
        "recent_sentence_counts": list(personality.phase.recent_sentence_counts),
    }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate sentence reshaping and rhythm modeling"""
    print("Layered Flow Personality Demo")
    print("=" * 70)
    print("\nTraits reshape sentence structure • Rhythm varies naturally\n")
    
    # Test conversations showing structure variation
    conversations = [
        ("Hey! What's up?", ""),  # Casual → playfulness↑, directness↓
        ("How do I use a for loop?", "You could use a for loop to iterate through items"),
        ("Should I manually copy all 500 files?", ""),  # Inefficient → resourcefulness↑
        ("This is so confusing and frustrating", ""),  # Frustration → patience↑, playfulness↓
        ("That makes sense, thanks!", ""),  # Gratitude → warmth↑
        ("What about error handling?", "You can add try-except blocks to handle errors gracefully"),
        ("Got it", ""),  # Brief input
        ("Can you explain that in more detail?", ""),  # Asking for detail → patience↑
    ]
    
    print("=" * 70)
    for i, (user_input, base_content) in enumerate(conversations, 1):
        print(f"\n[Turn {i}]")
        print(f"👤 User: {user_input}")
        
        response = process(user_input, base_content)
        phase = get_phase_info()
        
        print(f"🤖 AI: {response}")
        
        # Show structural influence
        traits = phase['traits']
        print(f"\n   Structure influence:")
        print(f"   • Directness: {traits['directness']:.2f} " + 
              ("(imperative)" if traits['directness'] > 0.85 else "(exploratory)" if traits['directness'] < 0.6 else "(balanced)"))
        print(f"   • Assertiveness: {traits['assertiveness']:.2f} " +
              ("(declarative)" if traits['assertiveness'] > 0.85 else "(suggestive)" if traits['assertiveness'] < 0.6 else "(moderate)"))
        print(f"   • Patience: {traits['patience']:.2f} " +
              ("(layered)" if traits['patience'] > 0.8 else "(stripped)" if traits['patience'] < 0.5 else "(natural)"))
        print(f"   • Warmth: {traits['warmth']:.2f} " +
              ("(collaborative)" if traits['warmth'] > 0.75 else "(instructional)" if traits['warmth'] < 0.5 else "(neutral)"))
        
        # Show rhythm
        print(f"\n   Rhythm:")
        print(f"   • Length: {len(response.split())} words")
        print(f"   • Target: {phase.get('rhythm_target', 'medium')}")
        if len(phase.get('recent_lengths', [])) >= 3:
            recent = list(phase['recent_lengths'])[-3:]
            print(f"   • Recent pattern: {recent}")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("  ✓ Traits reshape sentence structure (not just add texture)")
    print("  ✓ Directness: imperative vs exploratory forms")
    print("  ✓ Assertiveness: declarative vs suggestive")
    print("  ✓ Patience: concise vs layered explanations")
    print("  ✓ Warmth: collaborative vs instructional language")
    print("  ✓ Rhythm modeling: length varies to prevent monotony")
    print("  ✓ Natural pacing: brief ↔ medium ↔ detailed")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
