# FUNCTION REFERENCE GUIDE - WHAT EACH FUNCTION DOES

## 🎯 CORE ORCHESTRATION FUNCTIONS

### `chatbot_interface(user_message, chat_history, allow_web, max_code_tokens, search_mode)`
**Line 7172** | **Entry Point**
- Calls pdf_tutor_with_explicit_routing()
- Normalizes output format
- Appends to chat history
- Returns: (chat_history, confidence_score)

### `pdf_tutor_with_explicit_routing(query, chunks, sources, chunk_emb, emb_model, llm, allow_web, max_code_tokens, search_mode)`
**Line 5497** | **Main Orchestrator - THE BRAIN**
- Loads chat history from database
- Validates and cleans query
- Extracts username if present
- Handles greetings
- Detects explicit commands (search:, web search:)
- Checks web search permissions
- Routes to appropriate handler based on search_mode
- Builds prompts with context
- Generates LLM responses
- Humanizes output
- Saves to chat history
- Returns: (answer, confidence, sources, search_used)

---

## 🧠 INTENT DETECTION & CLASSIFICATION

### `ImprovedIntentClassifier.classify(query, chat_history)`
**Line 4036** | **Intent Detection Engine**
- Normalizes query text
- Checks for meta questions ("what did I ask?")
- Detects greetings
- Detects follow-ups using:
  - Confusion patterns
  - Explicit follow-up requests
  - Direct references
  - Pronouns (it, this, that)
  - Short responses after long AI messages
  - Continuation words (and, but, so)
  - Contextual questions
- Uses embeddings for topic similarity (cosine similarity)
- Returns: IntentResult with intent type + confidence + reasoning

### `classify_intent_fast(query, chat_history)`
**Line 4513** | **Quick Intent Wrapper**
- Wrapper around ImprovedIntentClassifier
- Returns string intent ("greeting", "followup", "question")

### `AutoModeRouter.decide(query, has_pdfs, allow_web)`
**Line 3534** | **Knowledge Source Router**
- Decides which knowledge source to use
- Checks query patterns for PDF references
- Checks for web search indicators
- Returns: SourceDecision (PDF, WEB, LLM_ONLY, HYBRID, AUTO)

---

## 📚 RAG (PDF/DOCUMENT) FUNCTIONS

### `process_pdfs(uploaded_files)`
**Line 6971** | **PDF Upload Handler**
- Extracts text from uploaded PDFs using PyPDF2
- Chunks text into 5-sentence segments with 1 overlap
- Generates embeddings using SentenceTransformer
- Stores chunks in FAISS index
- Saves metadata to SQLite
- Returns: success message

### `try_hybrid_rag_search(query, top_k=5)`
**Line 5119** | **Hybrid RAG Retrieval**
- Combines semantic search (embeddings) + keyword search (BM25)
- Retrieves top_k most relevant chunks
- Returns: List of RetrievalResult objects

### `enhanced_rag_retrieval(query, top_k, min_score, source_filter, date_filter)`
**Line 8036** | **Advanced RAG Retrieval**
- Filters by minimum relevance score
- Filters by source document
- Filters by date range
- Re-ranks results
- Returns: (context_string, result_list)

### `_handle_pdf_rag(query, llm)`
**Line 6173** | **PDF Query Handler**
- Retrieves relevant chunks from RAG
- Builds context with source citations
- Generates answer using LLM
- Formats response with sources
- Returns: (answer, confidence, sources)

### `clear_pdfs()`
**Line 7077** | **Clear RAG Index**
- Deletes all PDF files from storage
- Clears FAISS index
- Clears SQLite metadata
- Resets persistent data

---

## 🌐 WEB SEARCH FUNCTIONS

### `run_web_search(query, cached_results, allow_cache, max_results, validate)`
**Line 3334** | **Main Web Search**
- Checks cache first (7-day TTL)
- Calls fetch_duckduckgo_results()
- Validates results (spam detection)
- Saves to cache
- Returns: List[SearchResult]

### `fetch_duckduckgo_results(query, max_results, fetch_previews, timeout, max_retries)`
**Line 2925** | **DuckDuckGo Scraper**
- Sends POST request to DuckDuckGo Lite
- Parses HTML with BeautifulSoup
- Extracts titles, URLs, snippets
- Optionally fetches full page content
- Retries on failure
- Returns: List[(title, url, content)]

### `run_deep_web_search(query, llm, max_queries, max_results)`
**Line 3147** | **Multi-Query Deep Search**
- Expands query using LLM (generates 3-5 related queries)
- Runs search for each query
- Combines and deduplicates results
- Validates and ranks
- Returns: List[SearchResult]

### `run_smart_web_search(query, cached_results, max_results)`
**Line 3209** | **Smart Search with Enhancement**
- Spell checks query
- Suggests corrections
- Expands query if needed
- Fetches results
- Validates quality
- Returns: List[SearchResult]

### `validate_search_results(query, results, min_length, max_duplicates)`
**Line 2121** | **Result Quality Filter**
- Scores results based on:
  - Length (too short = spam)
  - Query term presence
  - Title relevance
  - Duplicate content
- Filters out low-quality results
- Returns: List[validated_results]

### `WebSearchFirewall.check_permission()`
**Line 557** | **Rate Limiting & Permission**
- Tracks search count per session
- Enforces max 5 searches
- Returns: (allowed: bool, error_message: str)

### `_handle_web_search(query, llm)`
**Line 6099** | **Web Search Handler**
- Runs web search
- Formats results as context
- Generates answer using LLM + web context
- Returns: (answer, confidence, sources, True)

---

## 💬 CONVERSATION & CONTEXT MANAGEMENT

### `build_conversation_context(chat_history, max_turns=4, verbose=True)`
**Line 4528** | **Build Q&A Context**
- Extracts last N exchanges from history
- Preserves code blocks completely
- Truncates text responses intelligently
- Formats as structured conversation
- Returns: formatted context string

### `build_universal_conversation_context(chat_history, max_chars=3000, max_turns=3)`
**Line 9295** | **Universal Context Builder**
- Handles ALL content types (code, text, tables)
- Preserves formatting
- Truncates to character limit
- Returns: conversation context string

### `build_code_aware_context(chat_history, max_chars=2500)`
**Line 4639** | **Code-Preserving Context**
- Detects code blocks in previous responses
- Preserves code completely
- Truncates only explanations
- Returns: code-aware context

### `enhance_system_for_followup(system_msg, query, context)`
**Line 9366** | **Follow-up System Enhancer**
- Detects what follow-up is about (code, explanation, etc.)
- Adds special instructions to system message
- Tells LLM to reference conversation context
- Returns: enhanced system message

### `build_memory_context_with_name(query=None)`
**Line 1567** | **User Memory Context**
- Gets user name and preferences
- Builds personalized context
- Includes interaction stats
- Returns: memory context string

---

## 🤖 LLM INTERACTION

### `build_prompt_with_format(system_msg, user_msg, context, model_format)`
**Line 4921** | **Format Prompt for Model**
- Detects intent (follow-up, meta, new question)
- Builds appropriate context
- Formats for specific model type:
  - llama3: `<|begin_of_text|>...`
  - mistral: `<s>[INST]...`
  - chatml: `<|im_start|>...`
- Returns: formatted prompt string

### `detect_model_format(model_path)`
**Line 4867** | **Auto-Detect Model Type**
- Parses model filename
- Detects: Llama 2/3, Mistral, Phi, etc.
- Returns: format string

### `process_llm_answer(llm_out, is_code=False)`
**Line 788** | **Clean LLM Output**
- Extracts text from model response
- Fixes numbered lists
- Removes concatenated lists
- Protects dates in lists
- Cleans response (removes robotic phrases)
- Returns: cleaned answer string

### `format_llm_response(text)`
**Line 763** | **Fix Response Formatting**
- Fixes concatenated numbered lists
- Preserves dates (Jan. 2024 → Jan 2024 temporarily)
- Splits lists properly
- Restores dates
- Returns: formatted text

### `ModelManager.load_model(model_key, n_ctx, n_gpu_layers)`
**Line 7454** | **Load GGUF Model**
- Loads model from disk
- Sets context window size
- Configures GPU layers
- Updates global llm instance
- Returns: success boolean

---

## 🎨 RESPONSE HUMANIZATION

### `humanize_response_sentient(text, user_msg, chat_history)`
**Line 8556** | **Main Humanizer - THE PERSONALITY**
- Handles special cases (greetings, thanks, success)
- Preserves code blocks
- Removes robotic language
- Makes conversational
- Adds natural fillers
- Adds thinking process
- Adds personality quirks
- Adds proactive suggestions
- Adds memory callbacks
- Adds contextual emojis (BUGGY!)
- Adds closers
- Returns: humanized response

### `remove_robotic_language(text)`
**Line 8859** | **Remove AI Phrases**
- Removes "As an AI language model..."
- Removes "I apologize for any confusion..."
- Removes "I don't have personal opinions..."
- Removes other robotic patterns
- Returns: more natural text

### `make_conversational(text)`
**Line 8927** | **Add Conversational Tone**
- Adds casual phrases
- Softens technical language
- Adds transitional words
- Returns: conversational text

### `add_natural_fillers(text, user_query)`
**Line 8970** | **Add Conversational Fillers**
- Adds "Actually", "Basically", "Essentially"
- Context-aware placement
- Random selection
- Returns: text with fillers

### `add_contextual_emojis(text, user_query)`
**Line 8416** | **Add Emojis (BUGGY!)**
- Splits into sentences
- Decides which sentences get emojis
- Chooses contextual emojis
- Max 3 emojis per response (TOO LOW!)
- Returns: text with emojis

### `EmojiPersonality.should_add_emoji(sentence_count, has_code)`
**Line 8389** | **Emoji Decision Logic**
- Max 3 emojis total
- 30% chance for code responses
- 40% chance for short responses
- 60% chance for normal responses
- Returns: boolean (add emoji or not)

### `add_thinking_process_with_emoji(answer, query)`
**Line 8705** | **Add Thinking**
- 20% chance (TOO LOW!)
- Adds "Hmm 🤔 let me think..."
- Only for long responses (30+ words)
- Returns: text with thinking

### `add_personality_quirks_with_emoji(answer, query)`
**Line 8721** | **Add Quirks**
- 15% chance (TOO LOW!)
- Adds context-aware quirks
- "(yeah, this one's mind-bendy 🤯)"
- Returns: text with quirks

### `add_proactive_suggestions_with_emoji(answer, query, chat_history)`
**Line 8753** | **Add Suggestions**
- 35% chance (LOW)
- "Want me to show the advanced version?"
- "I can walk you through each line"
- Context-aware
- Returns: text with suggestion

### `add_memory_callbacks_with_emoji(answer, query, chat_history)`
**Line 8792** | **Reference Past Conversations**
- 25% chance (LOW)
- Finds topic overlap with past
- "This relates to what we talked about earlier"
- Returns: text with callback

### `maybe_add_closer_with_emoji(text, user_query)`
**Line 8823** | **Add Closing Phrase**
- 25% chance (LOW!)
- "Make sense? 😊"
- "Does that help? ✨"
- Context-aware
- Returns: text with closer

---

## 👤 USER MANAGEMENT

### `UserManager.__init__(db_path)`
**Line 1078** | **User Manager Init**
- Connects to SQLite database
- Creates users table if not exists
- Loads all user profiles
- Sets current user to "default"

### `UserManager.create_user(user_id, name)`
**Line 1188** | **Create User Profile**
- Creates new user with default preferences
- Saves to database
- Returns: user_id

### `UserManager.get_user_data(key_path, default=None)`
**Line 1253** | **Get User Data**
- Navigates nested dictionary using dot notation
- "preferences.use_emojis" → user["preferences"]["use_emojis"]
- Returns: value or default

### `UserManager.set_user_data(key_path, value)`
**Line 1269** | **Update User Data**
- Updates nested dictionary
- Saves to database
- Example: set_user_data("name", "Alice")

### `UserManager.add_to_chat_history(user_msg, ai_msg)`
**Line 1342** | **Append to History**
- Adds exchange to chat_history list
- Includes timestamp
- Saves to database
- Limits to last 100 messages

### `build_user_context_for_prompt()`
**Line 1473** | **Build User Context**
- Gets user name, preferences, interests
- Formats as context string
- Used in system messages
- Returns: user context

### `extract_and_save_username(message)`
**Line 1600** | **Extract Name from Message**
- Detects "I'm NAME", "My name is NAME"
- Saves to user profile
- Returns: extracted name or None

---

## 🗄️ DATA PERSISTENCE

### `load_chat_history()`
**Line 1645** | **Load Chat from Disk**
- Loads from chat_history.json
- Returns: (chat_history_list, interaction_count)

### `save_chat_history(history, count)`
**Line 1656** | **Save Chat to Disk**
- Saves to chat_history.json
- Writes as formatted JSON

### `load_pdf_metadata()`
**Line 1673** | **Load PDF Index Metadata**
- Loads from pdf_metadata.json
- Returns: metadata dict

### `save_pdf_metadata(metadata)`
**Line 1684** | **Save PDF Metadata**
- Saves to pdf_metadata.json
- Tracks uploaded PDFs

### `load_search_cache()`
**Line 1627** | **Load Web Search Cache**
- Loads from web_cache.json
- Returns: cache dict

### `save_search_cache(cache)`
**Line 1637** | **Save Search Cache**
- Saves to web_cache.json
- 7-day expiry

---

## 🛠️ UTILITY FUNCTIONS

### `clean_response(text)`
**Line 4732** | **Clean Bad Patterns**
- Removes conversation starters
- Removes "How's it going?"
- Removes "Response:", "User:", etc.
- Returns: cleaned text

### `clean_response_safe(text)`
**Line 859** | **Safe Response Cleaner**
- Version with try-except
- Won't crash on errors
- Returns: cleaned text

### `cosine_sim(emb1, emb2)`
**Line 3859** | **Cosine Similarity**
- Calculates angle between vectors
- Used for semantic similarity
- Returns: float (0-1)

### `format_age(seconds)`
**Line 3322** | **Human-Readable Time**
- Converts seconds to "2 hours ago", "3 days ago"
- Returns: formatted string

### `get_file_hash(filepath)`
**Line 1693** | **Calculate File Hash**
- Uses hashlib.sha256
- Returns: hex digest string

### `debug_log(message)`
**Line 3877** | **Debug Logger**
- Prints [DEBUG] message if DEBUG_MODE=True
- Used throughout codebase for debugging

---

## 📊 STATISTICS & DIAGNOSTICS

### `get_cache_stats()`
**Line 3434** | **Web Cache Statistics**
- Returns cache size, entry count, hit rate

### `get_rag_stats()`
**Line 7724** | **RAG Index Statistics**
- Returns PDF count, chunk count, source count

### `view_cache_stats()`
**Line 7245** | **Display Cache Stats**
- Formats stats for UI display
- Returns: formatted string

### `diagnose_cache_system()`
**Line 7783** | **Cache System Diagnosis**
- Checks cache health
- Reports issues
- Returns: diagnostic report

---

## 🔧 MAINTENANCE FUNCTIONS

### `clear_history()`
**Line 7237** | **Clear Chat History**
- Empties chat_history
- Resets interaction count
- Saves to disk

### `clear_all_caches()`
**Line 3474** | **Clear All Caches**
- Clears web search cache
- Clears RAG cache
- Resets counters

### `clear_rag_cache()`
**Line 7730** | **Clear RAG Cache Only**
- Clears FAISS index cache
- Forces re-indexing

### `rebuild_rag_index()`
**Line 7737** | **Rebuild PDF Index**
- Re-indexes all PDFs
- Regenerates embeddings
- Rebuilds FAISS index

### `cleanup_database()`
**Line 7769** | **Clean Old Data**
- Removes old entries
- Compacts database
- Returns: cleanup report

---

## 📈 QUERY ENHANCEMENT

### `expand_query_smart(llm, query, max_queries=3)`
**Line 2240** | **Smart Query Expansion**
- Uses LLM to generate related queries
- Example: "python loops" → ["for loops python", "while loops", "loop iteration"]
- Returns: List[expanded_queries]

### `suggest_spell_corrections(query)`
**Line 2368** | **Spell Check**
- Detects common typos
- Suggests corrections
- Returns: List[corrected_queries]

### `normalize_query(query)`
**Line 2427** | **Query Normalization**
- Lowercases
- Removes extra whitespace
- Removes punctuation
- Returns: normalized query

### `extract_key_terms(query)`
**Line 8207** | **Extract Keywords**
- Uses RAKE algorithm
- Extracts important phrases
- Returns: List[key_terms]

### `rerank_results(query, results, key_terms)`
**Line 8229** | **Re-rank Results**
- Re-scores based on key terms
- Boosts results with more matches
- Returns: sorted results

---

## 🎯 SUMMARY

**Total Functions: 100+**

**Categories:**
- 🔀 Orchestration: 2 functions
- 🧠 Intent Detection: 3 functions  
- 📚 RAG/PDF: 10+ functions
- 🌐 Web Search: 10+ functions
- 💬 Conversation: 10+ functions
- 🤖 LLM: 10+ functions
- 🎨 Humanization: 20+ functions
- 👤 User Management: 10+ functions
- 🗄️ Data Persistence: 10+ functions
- 🛠️ Utilities: 10+ functions
- 📊 Statistics: 5+ functions
- 🔧 Maintenance: 5+ functions
- 📈 Query Enhancement: 5+ functions

**Most Critical Functions:**
1. `pdf_tutor_with_explicit_routing()` - Main brain
2. `ImprovedIntentClassifier.classify()` - Intent detection
3. `build_prompt_with_format()` - Prompt construction
4. `humanize_response_sentient()` - Personality
5. `build_universal_conversation_context()` - Context building
