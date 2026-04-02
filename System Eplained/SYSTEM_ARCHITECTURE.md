# YAN.PY SYSTEM ARCHITECTURE - COMPLETE GUIDE

## 📋 TABLE OF CONTENTS
1. [System Overview](#system-overview)
2. [Main Flow](#main-flow)
3. [Core Components](#core-components)
4. [Module Breakdown](#module-breakdown)
5. [Data Flow Diagram](#data-flow-diagram)

---

## 🏗️ SYSTEM OVERVIEW

**yan.py** is a 9,462-line AI assistant system with:
- **PDF/Document RAG** (Retrieval-Augmented Generation)
- **Web Search Integration** (DuckDuckGo)
- **Local LLM** (llama.cpp with GGUF models)
- **Conversation Memory & Context Management**
- **Multi-source Intelligence Routing**
- **User Personalization**

**Key Stats:**
- ~100+ functions
- ~20+ classes
- 7 major subsystems

---

## 🔄 MAIN FLOW

```
User Query
    ↓
chatbot_interface()
    ↓
pdf_tutor_with_explicit_routing()  ← MAIN ORCHESTRATOR
    ↓
┌─────────────────────────────────────────┐
│ 1. Load chat history                    │
│ 2. Validate query                       │
│ 3. Check for username/greeting          │
│ 4. Detect explicit commands             │
│ 5. Route to appropriate handler         │
│ 6. Build prompt with context            │
│ 7. Generate response with LLM           │
│ 8. Humanize response                    │
│ 9. Save to history                      │
│ 10. Return to user                      │
└─────────────────────────────────────────┘
```

---

## 🧩 CORE COMPONENTS

### 1. **Entry Point**
- `chatbot_interface()` (Line 7172)
  - Gradio UI wrapper
  - Normalizes output from pdf_tutor
  - Returns chat history + confidence score

### 2. **Main Orchestrator**
- `pdf_tutor_with_explicit_routing()` (Line 5497)
  - **THE BRAIN** of the entire system
  - Handles all query types
  - Routes to appropriate subsystems
  - Returns: (answer, confidence, sources, search_used)

### 3. **Knowledge Sources**
Three main knowledge sources:
1. **RAG (PDF Documents)** - Local knowledge base
2. **Web Search** - Real-time internet data
3. **LLM Only** - Model's training knowledge

---

## 📦 MODULE BREAKDOWN

### 🧠 MODULE 1: INTENT DETECTION & ROUTING

#### **Components:**
- `ImprovedIntentClassifier` (Line 3921)
- `classify_intent_fast()` (Line 4513)
- `AutoModeRouter` (Line 3534)

#### **Responsibilities:**
1. **Classify user intent:**
   - GREETING
   - QUESTION
   - FOLLOWUP
   - META_QUESTION
   - CODE_REQUEST
   - WEB_SEARCH

2. **Determine knowledge source:**
   - Should use PDFs?
   - Should search web?
   - Should use LLM only?

#### **How it works:**
```python
# Step 1: Intent Classification
intent_result = INTENT_CLASSIFIER.classify(query, chat_history)

# Step 2: Source Routing
if intent == "followup":
    # Use conversation context
    context = build_conversation_context(chat_history)
elif "in the pdf" in query:
    # Use RAG
    handler = _handle_pdf_rag()
elif "search the web" in query:
    # Use web search
    handler = _handle_web_search()
else:
    # Auto-route based on query type
    decision = AUTO_ROUTER.decide(query, has_pdfs, allow_web)
```

**Key Functions:**
- Pattern matching (pronouns, connectors, confusion words)
- Embedding-based similarity (cosine similarity with previous message)
- Confidence scoring (combines rule-based + embeddings)

---

### 📚 MODULE 2: RAG (RETRIEVAL-AUGMENTED GENERATION)

#### **Components:**
- `HybridRAGManager` (External import)
- `AdvancedRAGEngine` (Line 56-62)
- `RAGKnowledgeDatabase` (External import)

#### **Responsibilities:**
1. **PDF Processing:**
   - Extract text from PDFs
   - Chunk into manageable pieces
   - Generate embeddings
   - Store in vector database (FAISS + SQLite)

2. **Retrieval:**
   - Semantic search using embeddings
   - Keyword search using BM25
   - Hybrid ranking (combines both)
   - Re-ranking for relevance

3. **Context Building:**
   - Filter results by score threshold
   - Format with source citations
   - Inject into LLM prompt

#### **Key Functions:**
- `process_pdfs()` (Line 6971) - Upload and index PDFs
- `try_hybrid_rag_search()` (Line 5119) - Hybrid retrieval
- `enhanced_rag_retrieval()` (Line 8036) - Advanced retrieval with filters
- `_handle_pdf_rag()` (Line 6173) - Main PDF query handler

#### **Flow:**
```python
User uploads PDF
    ↓
process_pdfs()
    ↓
Extract text (PyPDF2)
    ↓
Chunk text (5 sentences, 1 overlap)
    ↓
Generate embeddings (SentenceTransformer)
    ↓
Store in FAISS index + SQLite DB
    ↓
User asks question
    ↓
Embed query
    ↓
Hybrid search (semantic + keyword)
    ↓
Retrieve top 5 chunks
    ↓
Build context with sources
    ↓
LLM generates answer
```

---

### 🌐 MODULE 3: WEB SEARCH

#### **Components:**
- `WebSearchFirewall` (Line 557)
- `WebSearchPermissionManager` (Line 146)
- `fetch_duckduckgo_results()` (Line 2925)
- `run_smart_web_search()` (Line 3209)

#### **Responsibilities:**
1. **Search Execution:**
   - Query DuckDuckGo Lite API
   - Parse HTML results
   - Extract titles, URLs, snippets

2. **Result Enhancement:**
   - Fetch full page content (optional)
   - Validate result quality
   - Rank by relevance

3. **Safety & Rate Limiting:**
   - Permission system (user must approve)
   - Firewall (max 5 searches per session)
   - Cache results (7-day TTL)

4. **Query Enhancement:**
   - Spell correction
   - Query expansion (generate related queries)
   - Entity extraction

#### **Key Functions:**
- `run_web_search()` (Line 3334) - Main entry point
- `run_deep_web_search()` (Line 3147) - Multi-query search
- `validate_search_results()` (Line 2121) - Quality scoring
- `_handle_web_search()` (Line 6099) - Handler in routing system

#### **Flow:**
```python
Detect web search need
    ↓
Check permissions (WebSearchFirewall)
    ↓
Ask user for approval (if first time)
    ↓
Check cache (7-day expiry)
    ↓
If not cached:
    ↓
    Enhance query (spell check, expansion)
    ↓
    Fetch DuckDuckGo results (max 6)
    ↓
    Validate quality (spam detection)
    ↓
    Optionally fetch full page content
    ↓
    Save to cache
    ↓
Format results as context
    ↓
Inject into LLM prompt
    ↓
Generate answer with citations
```

---

### 💬 MODULE 4: CONVERSATION MANAGEMENT

#### **Components:**
- `UserManager` (Line 1078)
- `build_conversation_context()` (Line 4528)
- `build_universal_conversation_context()` (Line 9295)
- `build_code_aware_context()` (Line 4639)

#### **Responsibilities:**
1. **Chat History:**
   - Store all Q&A pairs
   - Load from SQLite database
   - Support multi-turn conversations

2. **Context Building:**
   - Extract last N exchanges
   - Preserve code blocks fully
   - Truncate text intelligently
   - Handle different content types

3. **Follow-up Detection:**
   - Detect pronouns (it, this, that)
   - Detect connectors (also, and, but)
   - Detect confusion (don't understand)
   - Use embeddings for topic continuity

4. **Memory Context:**
   - User preferences
   - Past topics discussed
   - Learning goals
   - Skill assessments

#### **Key Functions:**
- `build_conversation_context()` (Line 4528) - Build Q&A context
- `build_code_aware_context()` (Line 4639) - Preserve code blocks
- `enhance_system_for_followup()` (Line 9366) - Enhance system message
- `build_memory_context_with_name()` (Line 1567) - User personalization

#### **Flow:**
```python
User asks follow-up: "what does it do?"
    ↓
classify_intent() detects FOLLOWUP (0.92 confidence)
    ↓
build_prompt() checks intent.needs_context == True
    ↓
build_universal_conversation_context()
    ↓
Extract last 3 exchanges from chat_history
    ↓
For each exchange:
    - Preserve code blocks fully
    - Truncate text responses to 500 chars
    - Format as "User: ... Assistant: ..."
    ↓
Combine into context string (max 3000 chars)
    ↓
enhance_system_for_followup()
    ↓
Add special instruction:
    "This is a follow-up about CODE. The complete code
     is in the conversation history. Reference it directly."
    ↓
Build final prompt:
    System: [enhanced system message]
    Context: [conversation history with code]
    Query: "what does it do?"
    ↓
LLM understands "it" = the code from context
    ↓
Generate explanation
```

---

### 🤖 MODULE 5: LLM INTERACTION

#### **Components:**
- `ModelManager` (Line 7368)
- `build_prompt_with_format()` (Line 4921)
- `process_llm_answer()` (Line 788)

#### **Responsibilities:**
1. **Model Management:**
   - Scan for .gguf models
   - Load/unload models
   - Detect model format (Llama3, Mistral, Phi, etc.)
   - GPU offloading support

2. **Prompt Building:**
   - Format for specific model types
   - Inject system message
   - Inject context (PDFs, web, conversation)
   - Handle different chat templates

3. **Response Processing:**
   - Extract text from model output
   - Clean formatting issues
   - Fix numbered lists
   - Remove prompt leakage

#### **Supported Formats:**
- `llama3` - Llama 3 / 3.1 / 3.2
- `llama2` - Llama 2, CodeLlama
- `mistral-instruct` - Mistral 7B Instruct
- `chatml` - Generic ChatML format
- `phi` - Microsoft Phi models

#### **Key Functions:**
- `build_prompt_with_format()` (Line 4921) - Format prompt for model
- `detect_model_format()` (Line 4867) - Auto-detect model type
- `process_llm_answer()` (Line 788) - Clean LLM output
- `format_llm_response()` (Line 763) - Fix formatting

#### **Prompt Template Examples:**

**Llama 3:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{context}

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**Mistral:**
```
<s>[INST] {system_message}

{context}

{query} [/INST]
```

**ChatML:**
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{context}

{query}<|im_end|>
<|im_start|>assistant
```

---

### 🎨 MODULE 6: RESPONSE HUMANIZATION

#### **Components:**
- `humanize_response_sentient()` (Line 8556)
- `EmojiPersonality` (Line 8277)
- 20+ humanization functions

#### **Responsibilities:**
1. **Make responses natural:**
   - Remove robotic phrases ("As an AI...", "I apologize...")
   - Add conversational fillers ("Hmm", "Actually", "Basically")
   - Vary sentence starters
   - Add natural flow

2. **Add personality:**
   - Contextual emojis (but BUGGY - too few!)
   - Thinking process ("Hmm 🤔 let me think...")
   - Personality quirks ("yeah, this one's mind-bendy 🤯")
   - Proactive suggestions ("Want me to show the advanced version?")

3. **Handle different content:**
   - Preserve code blocks
   - Format lists properly
   - Add closers ("Make sense? 😊")
   - Reference previous conversations

#### **Key Functions:**
- `humanize_response_sentient()` (Line 8556) - Main humanizer
- `remove_robotic_language()` (Line 8859) - Remove AI phrases
- `make_conversational()` (Line 8927) - Add natural language
- `add_contextual_emojis()` (Line 8416) - Add emojis (BUGGY!)
- `humanize_code_response_with_emojis()` (Line 8667) - Handle code

#### **Pipeline:**
```python
LLM raw output:
"As an AI language model, I can help you with that. 
Regarding your question, the answer is..."

    ↓
remove_robotic_language()
"I can help you with that. The answer is..."

    ↓
make_conversational()
"Sure! The answer is actually..."

    ↓
add_natural_fillers()
"So basically, the answer is actually..."

    ↓
add_thinking_process_with_emoji()
"Hmm 🤔 let me think... So basically, the answer is..."

    ↓
add_contextual_emojis()
"Hmm 🤔 let me think... So basically, the answer is... ✨"

    ↓
maybe_add_closer_with_emoji()
"Hmm 🤔 let me think... So basically, the answer is... ✨ Make sense? 😊"
```

---

### 👤 MODULE 7: USER MANAGEMENT

#### **Components:**
- `UserManager` (Line 1078)
- SQLite database (`users.db`)

#### **Responsibilities:**
1. **User Profiles:**
   - Store user name, preferences, interests
   - Track interaction statistics
   - Save learning goals
   - Skill assessments

2. **Preferences:**
   - Response style (concise/balanced/detailed)
   - Code style (minimal/commented/verbose)
   - Expertise level (beginner/intermediate/advanced)
   - Emoji usage (on/off) - **CURRENTLY NOT USED!**

3. **Session Management:**
   - Create/switch users
   - Auto-create "default" user
   - Load/save user data to SQLite

4. **Context Enhancement:**
   - Add user name to responses
   - Personalize based on expertise level
   - Reference past interactions

#### **Key Functions:**
- `UserManager.__init__()` (Line 1078) - Initialize manager
- `create_user()` - Create new user profile
- `get_user_data()` - Get any user data field
- `set_user_data()` - Update user data field
- `add_to_chat_history()` - Append to chat history
- `build_user_context_for_prompt()` (Line 1473) - Build context

#### **Database Schema:**
```python
User Profile Structure:
{
    "user_id": "default",
    "name": "Alice",
    "created_at": "2024-01-15T10:30:00",
    
    "preferences": {
        "response_style": "balanced",
        "code_style": "commented",
        "expertise_level": "intermediate",
        "preferred_language": "python",
        "use_emojis": True,  # ← NOT CHECKED! (BUG)
        "stream_responses": True
    },
    
    "interests": ["machine learning", "web development"],
    "learning_goals": ["learn React", "master Python"],
    
    "chat_history": [
        {"user": "...", "ai": "...", "timestamp": ...},
        {"user": "...", "ai": "...", "timestamp": ...}
    ],
    
    "interaction_stats": {
        "total_questions": 42,
        "topics_discussed": {...},
        "favorite_topics": [...]
    }
}
```

---

## 🔀 DATA FLOW DIAGRAM

### Complete Request Flow:

```
┌─────────────────────────────────────────────────────────────┐
│                          USER QUERY                          │
│                   "explain this code"                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼────────────┐
                │  chatbot_interface()   │
                └───────────┬────────────┘
                            │
        ┌───────────────────▼───────────────────┐
        │ pdf_tutor_with_explicit_routing()     │
        │                                       │
        │  STEP 1: Load chat history from DB   │
        │  STEP 2: Validate query              │
        │  STEP 3: Check username/greeting     │
        │  STEP 4: Detect explicit commands    │
        │  STEP 5: Route to handler            │
        └───────────────────┬───────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │   Intent Classification        │
            │   (ImprovedIntentClassifier)   │
            │                                │
            │  - Rule-based patterns         │
            │  - Embedding similarity        │
            │  - Confidence scoring          │
            └───────────────┬────────────────┘
                            │
                ┌───────────▼────────────┐
                │   FOLLOWUP detected    │
                │   Confidence: 0.92     │
                │   needs_context: True  │
                └───────────┬────────────┘
                            │
        ┌───────────────────▼────────────────────┐
        │   build_conversation_context()         │
        │                                        │
        │  Extract last 3 exchanges:             │
        │  - User: "write bubble sort"           │
        │  - AI: [code here]                     │
        │  - User: "explain this code" ← current │
        └───────────────────┬────────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │   build_prompt_with_format()   │
            │                                │
            │  System: You are a helpful AI │
            │          This is a FOLLOWUP.   │
            │          Code is in context.   │
            │                                │
            │  Context: [conversation above] │
            │                                │
            │  Query: explain this code      │
            └───────────────┬────────────────┘
                            │
                ┌───────────▼────────────┐
                │   LLM Generation       │
                │   (llama.cpp)          │
                │                        │
                │   Model: Llama 3.1     │
                │   Temperature: 0.7     │
                │   Max tokens: 800      │
                └───────────┬────────────┘
                            │
            ┌───────────────▼─────────────────┐
            │   process_llm_answer()          │
            │                                 │
            │  - Extract text                 │
            │  - Fix formatting               │
            │  - Clean numbered lists         │
            │  - Remove prompt leakage        │
            └───────────────┬─────────────────┘
                            │
        ┌───────────────────▼──────────────────────┐
        │   humanize_response_sentient()           │
        │                                          │
        │  - Remove "As an AI..."                  │
        │  - Add conversational fillers            │
        │  - Add emojis (BUGGY - too few!)         │
        │  - Add closers                           │
        └───────────────────┬──────────────────────┘
                            │
            ┌───────────────▼────────────────┐
            │   Save to chat_history         │
            │   Save to SQLite DB            │
            └───────────────┬────────────────┘
                            │
                ┌───────────▼────────────┐
                │   Return to User       │
                │                        │
                │   "Sure! This code..." │
                │   Confidence: 0.85     │
                └────────────────────────┘
```

---

## 🎯 KEY SYSTEM CHARACTERISTICS

### 1. **Hybrid Intelligence**
- Combines 3 knowledge sources: PDFs + Web + LLM
- Auto-routing based on query type
- Confidence scoring for each source

### 2. **Contextual Awareness**
- Tracks conversation history
- Detects follow-ups with high accuracy
- Preserves code blocks in context
- References previous discussions

### 3. **Safety & Control**
- Web search requires permission
- Firewall limits (5 searches/session)
- Caching prevents redundant searches
- Result validation prevents spam

### 4. **Personalization**
- User profiles with preferences
- Name recognition and usage
- Expertise level adaptation
- Interest tracking

### 5. **Multi-Model Support**
- Works with any GGUF model
- Auto-detects model format
- Supports GPU acceleration
- Hot-swappable models

---

## 🐛 KNOWN ISSUES (From Our Fixes)

### Issue 1: META_QUESTION Bug ✅ FIXED
- **Problem:** "what are we talking about?" got no context
- **Location:** Line 9207-9214
- **Fix:** Treat META_QUESTION like FOLLOWUP

### Issue 2: Emoji Underuse ⚠️ TO FIX
- **Problem:** Very few emojis in responses
- **Root causes:**
  - use_emojis preference not checked
  - Low probability thresholds (15-35%)
  - Max 3 emojis per response
  - Early returns skip emoji functions
- **Fix:** 11 line changes needed

---

## 📊 SYSTEM STATISTICS

**Code Metrics:**
- Total lines: 9,462
- Functions: ~100+
- Classes: ~20+
- External dependencies: 15+

**Key Dependencies:**
- `llama-cpp-python` - LLM inference
- `sentence-transformers` - Embeddings
- `PyPDF2` - PDF parsing
- `gradio` - Web UI
- `sqlite3` - Data storage
- `faiss` - Vector search
- `requests` - Web scraping
- `BeautifulSoup` - HTML parsing

**Performance:**
- PDF indexing: ~1-2s per document
- Web search: ~2-5s per query
- LLM generation: ~5-30s (depends on model)
- Follow-up detection: ~0.1s

---

## 🎓 SUMMARY

The system is a **sophisticated multi-modal AI assistant** that:

1. **Understands context** through conversation tracking
2. **Retrieves knowledge** from PDFs, web, and LLM
3. **Routes intelligently** based on query type
4. **Generates naturally** with humanization
5. **Personalizes** based on user preferences

**Core Flow:** Query → Intent Detection → Source Routing → Context Building → LLM Generation → Humanization → Response

**Key Innovation:** Hybrid RAG + Web + LLM with automatic routing and context-aware follow-ups.

**Strengths:**
- Multi-source intelligence
- Context preservation
- Follow-up detection
- Personalization

**Weaknesses:**
- Emoji system buggy
- Some early returns skip processing
- Complex codebase (9k+ lines)
- Performance varies with model
