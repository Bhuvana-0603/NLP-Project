import streamlit as st
import language_tool_python
import spacy
import pyinflect
import pandas as pd 
import re 
import random 

#Initialize NLP Tools (cached for performance)

@st.cache_resource
def get_language_tool():
    try:
        tool = language_tool_python.LanguageTool('en-US')
        return tool
    except Exception as e:
        st.error(f"Error initializing LanguageTool. Is Java installed and in PATH? Error: {e}")
        return None

@st.cache_resource
def get_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError as e:
        st.error(f"SpaCy model not found. Please run: `python -m spacy download en_core_web_sm`. Error: {e}")
        return None

#Grammar Correction Function 
def correct_grammar(text, tool):
    if not tool: return text, 0
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches), len(matches)

#Tense Change Function (Rule-Based - with limitations)
def change_tense_rule_based(text, nlp_model, target_tense):
    if not nlp_model: return "spaCy model not loaded."
    doc = nlp_model(text)
    new_tokens = []
    verb_indices_and_forms = []

    for i, token in enumerate(doc):
        original_form = token.text
        new_form = token.text
        is_verb_or_aux = token.pos_ in ("VERB", "AUX")

        if is_verb_or_aux:
            if target_tense == "Past":
                new_form = token._.inflect("VBD") or token.lemma_
            elif target_tense == "Present":
                # Try VBZ (3rd person singular), then VBP (non-3rd person singular), then lemma
                new_form = token._.inflect("VBZ") or token._.inflect("VBP") or token.lemma_
            elif target_tense == "Future":
                new_form = token.lemma_ # Base form for future

            if new_form:
                 verb_indices_and_forms.append({"index": i, "original": original_form, "new": new_form, "pos": token.pos_, "dep": token.dep_, "lemma": token.lemma_})
            else: # Fallback if no new form generated
                verb_indices_and_forms.append({"index": i, "original": original_form, "new": original_form, "pos": token.pos_, "dep": token.dep_, "lemma": token.lemma_})

    future_will_added_for_main_verb = False
    # Reconstruct sentence, adding "will" for future tense if needed
    if target_tense == "Future":
       
        main_verb_candidate_indices = [
            v_info["index"] for v_info in verb_indices_and_forms
            if v_info["dep"] == "ROOT" and v_info["lemma"] not in ["be", "have"] 
        ]
        # If no ROOT verb, or ROOT is be/have, consider the first VERB not be/have
        if not main_verb_candidate_indices and verb_indices_and_forms:
             first_verb_info = verb_indices_and_forms[0]
             if first_verb_info["pos"] == "VERB" and first_verb_info["lemma"] not in ["be", "have"]:
                 main_verb_candidate_indices.append(first_verb_info["index"])
             elif not main_verb_candidate_indices and verb_indices_and_forms: 
                 main_verb_candidate_indices.append(verb_indices_and_forms[0]["index"])


        for i, token in enumerate(doc):
            verb_info = next((v for v in verb_indices_and_forms if v["index"] == i), None)

            if target_tense == "Future" and \
               not future_will_added_for_main_verb and \
               i in main_verb_candidate_indices:
                # Check if 'will' is already there (e.g. from original sentence or a previous aux)
                is_prev_token_will = new_tokens and new_tokens[-1].lower() == "will"
                # Check if current token is 'will' (e.g. from a modal)
                is_current_token_will_modal = token.lemma_ == "will" and token.pos_ == "AUX"

                if not is_prev_token_will and not is_current_token_will_modal:
                    new_tokens.append("will")
                    future_will_added_for_main_verb = True

            if verb_info:
                new_tokens.append(verb_info["new"])
            else:
                new_tokens.append(token.text)
    else: # Past or Present tense
        for i, token in enumerate(doc):
            verb_info = next((v for v in verb_indices_and_forms if v["index"] == i), None)
            if verb_info:
                new_tokens.append(verb_info["new"])
            else:
                new_tokens.append(token.text)

    result = " ".join(new_tokens)
    # Post-processing for common spacing issues
    result = result.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    result = result.replace(" 's", "'s").replace(" 're", "'re").replace(" 've", "'ve").replace(" n't", "n't")
    return result.strip()

#Part-of-Speech Tagging Function 
def get_pos_tags(text, nlp_model):
    if not nlp_model:
        return pd.DataFrame(columns=["Token", "POS Tag", "Fine-grained Tag", "Explanation"])
    doc = nlp_model(text)
    pos_data = []
    for token in doc:
        pos_data.append({
            "Token": token.text,
            "POS Tag": token.pos_,
            "Fine-grained Tag": token.tag_,
            "Explanation": spacy.explain(token.tag_)
        })
    return pd.DataFrame(pos_data)

#Tone Change Function (Rule-Based)

CONTRACTION_MAP_EXPAND = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
    "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would", "there'd've": "there would have",
    "there's": "there is", "they'd": "they would", "they'd've": "they would have",
    "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
    "where's": "where is", "where've": "where have", "who'll": "who will",
    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

INFORMAL_TO_FORMAL_WORDS = {
    r"\b(gonna)\b": "going to", r"\b(wanna)\b": "want to", r"\b(gotta)\b": "have to",
    r"\b(kinda)\b": "kind of", r"\b(sorta)\b": "sort of", r"\b(lemme)\b": "let me",
    r"\b(kids)\b": "children", r"\b(stuff)\b": "items", r"\b(a lot of)\b": "many",
    r"\b(awesome)\b": "excellent", r"\b(cool)\b": "impressive", r"\b(guy)\b": "man/person",
    r"\b(folks)\b": "people", r"\b(cuz|'cause)\b": "because", r"\b(ASAP)\b": "as soon as possible",
    r"\b(btw)\b": "by the way", r"\b(aka)\b": "also known as", r"\b(idk)\b": "I do not know",
    r"\b(tbh)\b": "to be honest", r"\b(ikr)\b": "I know, right"
}

FORMAL_TO_FRIENDLY_WORDS = {
    # Word replacements
    r"\b(utilize)\b": "use", r"\b(ascertain)\b": "find out", r"\b(commence)\b": "start",
    r"\b(endeavor)\b": "try", r"\b(procure)\b": "get", r"\b(terminate)\b": "end",
    r"\b(inform)\b": "tell", r"\b(require)\b": "need", r"\b(request)\b": "ask for",
    r"\b(subsequently)\b": "later", r"\b(therefore)\b": "so", r"\b(however)\b": "but",
    r"\b(furthermore)\b": "also",  r"\b(nevertheless)\b": "still",
    r"\b(individual)\b": "person", r"\b(children)\b": "kids", r"\b(residence)\b": "home",
    r"\b(sufficient)\b": "enough", r"\b(numerous)\b": "many", r"\b(approximately)\b": "about",
    # Contraction creation (from expanded forms)
    r"\b(is not)\b": "isn't", r"\b(are not)\b": "aren't", r"\b(cannot)\b": "can't",
    r"\b(can not)\b": "can't", # Common alternative spelling
    r"\b(will not)\b": "won't", r"\b(do not)\b": "don't", r"\b(does not)\b": "doesn't",
    r"\b(did not)\b": "didn't", r"\b(have not)\b": "haven't", r"\b(has not)\b": "hasn't",
    r"\b(had not)\b": "hadn't", r"\b(would not)\b": "wouldn't", r"\b(should not)\b": "shouldn't",
    r"\b(could not)\b": "couldn't", r"\b(I am)\b": "I'm", r"\b(you are)\b": "you're",
    r"\b(he is)\b": "he's", r"\b(she is)\b": "she's", r"\b(it is)\b": "it's",
    r"\b(we are)\b": "we're", r"\b(they are)\b": "they're", r"\b(I will)\b": "I'll",
    r"\b(you will)\b": "you'll", r"\b(he will)\b": "he'll", r"\b(she will)\b": "she'll",
    r"\b(it will)\b": "it'll", r"\b(we will)\b": "we'll", r"\b(they will)\b": "they'll",
}

GEN_Z_REPLACEMENTS = {
    r"\b(very)\b": "so", r"\b(really)\b": "lowkey", 
    r"\b(good|great|excellent)\b": "fire", 
    r"\b(funny|hilarious)\b": "sending me", 
    r"\b(friend)\b": "bro",
    r"\b(yes|okay|agreed)\b": "bet",
    r"\b(no)\b": "nah",
    r"\b(awesome)\b": "slay",
    r"\b(laughing out loud|lol)\b": "💀",
    r"\b(for real)\b": "fr",
    r"\b(to be honest)\b": "tbh",
    r"\b(in my opinion)\b": "imo",
    r"\b(not gonna lie)\b": "ngl",
}
GEN_Z_APPENDAGES = ["fr", "ngl", "no cap", "tbh", "periodt.", "slayyy"]
GEN_Z_EMOJIS = ["✨", "💅", "🔥", "💀", "😂", "💯", "👀"]

def expand_contractions_util(text, cmap=CONTRACTION_MAP_EXPAND):
    contractions_re = re.compile('(%s)' % '|'.join(sorted(cmap.keys(), key=len, reverse=True)), flags=re.IGNORECASE)
    def replace(match):
        val = match.group(0)
        expanded = cmap.get(val.lower())
        if expanded:
            return expanded[0].upper() + expanded[1:] if val[0].isupper() and len(expanded) > 1 else expanded
        return val
    return contractions_re.sub(replace, text)

def change_tone_simplified(text, target_tone):
    modified_text = text

    if target_tone == "Formal":
        modified_text = expand_contractions_util(modified_text)
        for informal, formal in INFORMAL_TO_FORMAL_WORDS.items():
            modified_text = re.sub(informal, formal, modified_text, flags=re.IGNORECASE)
        modified_text = modified_text.replace(" kinda ", " somewhat ").replace(" sorta ", " somewhat ") # More formal
        modified_text = re.sub(r"\b(right)\?$", ", is that correct?", modified_text, flags=re.IGNORECASE) # e.g. "..., right?"
        modified_text = re.sub(r"\b(ok|okay)\b", "alright", modified_text, flags=re.IGNORECASE)


    elif target_tone == "Friendly":
        for formal, friendly in FORMAL_TO_FRIENDLY_WORDS.items():
            modified_text = re.sub(formal, friendly, modified_text, flags=re.IGNORECASE)
        # Add some positive softeners
        if random.random() < 0.2 and "please" not in modified_text.lower():
            modified_text += " please?" if modified_text.endswith("?") else ", please."
        if random.random() < 0.25 and "thank you" not in modified_text.lower() and "thanks" not in modified_text.lower():
             modified_text = modified_text.strip().rstrip('.')
             modified_text += ", thanks!"

    elif target_tone == "Gen Z (Experimental)":
        # Apply word/phrase replacements
        for pattern, replacement in GEN_Z_REPLACEMENTS.items():
            modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)

        # Lowercase some parts (optional, more stylistic)
        if random.random() < 0.3: # 30% chance
            words = modified_text.split()
            if len(words) > 3: # Only for longer sentences
                # lowercase a few random words that are not the first word
                for _ in range(random.randint(1,min(3, len(words)//3))):
                    idx_to_lower = random.randint(1, len(words)-1)
                    if words[idx_to_lower].lower() not in ["i", "i'm", "i'll", "i'd", "i've"]: # Avoid I
                         words[idx_to_lower] = words[idx_to_lower].lower()
                modified_text = " ".join(words)


        # Append a common slang term or emoji if not already present
        if not any(term in modified_text.lower() for term in GEN_Z_APPENDAGES + GEN_Z_EMOJIS):
            if random.random() < 0.5: # 50% chance to append something
                choice = random.choice(GEN_Z_APPENDAGES + GEN_Z_EMOJIS)
                # Punctuation handling
                current_ends_with_punct = modified_text.strip()[-1] in ".!?" if modified_text.strip() else False
                punct_to_add = ""
                if current_ends_with_punct:
                    punct_to_add = modified_text.strip()[-1]
                    modified_text = modified_text.strip()[:-1]

                if choice in GEN_Z_EMOJIS:
                    modified_text = modified_text.strip() + " " + choice + punct_to_add
                else: # It's an appendage phrase
                    modified_text = modified_text.strip() + ", " + choice + punct_to_add
            elif random.random() < 0.3: # smaller chance to just add a random emoji
                modified_text = modified_text.strip() + " " + random.choice(GEN_Z_EMOJIS)


    return modified_text.strip()


# --- Streamlit App Interface ---
st.set_page_config(layout="wide", page_title="Advanced Text Processor")
st.title("📝 Advanced Text Processor")
st.markdown("Correct grammar, change tense, adjust tone, and analyze sentence structure.")

# Initialize tools
grammar_tool = get_language_tool()
spacy_nlp = get_spacy_model()

# Session State Initialization
for key in ['original_sentence', 'corrected_sentence', 'tense_changed_sentence', 'tone_changed_sentence',
            'last_processed_for_analysis', 'processed_input_sentence', 'pos_tags_df',
            'selected_tense', 'selected_tone']:
    if key not in st.session_state:
        if key == 'pos_tags_df':
            st.session_state[key] = pd.DataFrame()
        elif key in ['selected_tense', 'selected_tone']:
            st.session_state[key] = None # Or a default value
        else:
            st.session_state[key] = ""

if not st.session_state.original_sentence: # Default example
    st.session_state.original_sentence = "She go to the park and play yestaday. It be fun. We gonna have a lot of stuff to do, lol."

input_sentence = st.text_area(
    "Enter sentence(s):",
    height=100,
    key="user_input",
    value=st.session_state.original_sentence
)

if st.button("✨ Process Sentence", type="primary") or \
   (input_sentence and input_sentence != st.session_state.processed_input_sentence):
    st.session_state.original_sentence = input_sentence
    st.session_state.processed_input_sentence = input_sentence
    st.session_state.pos_tags_df = pd.DataFrame()
    st.session_state.tense_changed_sentence = ""
    st.session_state.tone_changed_sentence = ""
    st.session_state.selected_tense = None
    st.session_state.selected_tone = None


    if input_sentence:
        if grammar_tool:
            with st.spinner("Correcting grammar..."):
                corrected_sent, num_errors = correct_grammar(input_sentence, grammar_tool)
            st.session_state.corrected_sentence = corrected_sent
            st.session_state.last_processed_for_analysis = corrected_sent

            if spacy_nlp:
                with st.spinner("Generating POS tags..."):
                    st.session_state.pos_tags_df = get_pos_tags(corrected_sent, spacy_nlp)
            else:
                st.warning("spaCy model not loaded. POS tagging disabled.")
        else:
            st.session_state.corrected_sentence = input_sentence # Use original if grammar tool failed
            st.session_state.last_processed_for_analysis = input_sentence
            if spacy_nlp:
                 with st.spinner("Generating POS tags for original sentence..."):
                    st.session_state.pos_tags_df = get_pos_tags(input_sentence, spacy_nlp)
            else:
                st.warning("spaCy model not loaded. POS tagging disabled.")
    else: # No input
        st.session_state.original_sentence = ""
        st.session_state.corrected_sentence = ""
        st.session_state.last_processed_for_analysis = ""
        st.session_state.tense_changed_sentence = ""
        st.session_state.tone_changed_sentence = ""
        st.session_state.pos_tags_df = pd.DataFrame()


if st.session_state.corrected_sentence:
    st.subheader("✔️ Grammatically Corrected:")
    st.markdown(f"> {st.session_state.corrected_sentence}")
    if grammar_tool:
        original_matches = grammar_tool.check(st.session_state.original_sentence)
        st.caption(f"Found and corrected {len(original_matches)} potential error(s).")
    st.divider()

# --- Operations and Analysis on the Current Base Sentence ---
if st.session_state.last_processed_for_analysis:
    base_sentence_for_ops = st.session_state.last_processed_for_analysis
    st.info(f"Operating on / Analyzing: *'{base_sentence_for_ops}'*")

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        # --- Tense Change ---
        st.subheader("🔄 Change Tense (Rule-Based)")
        st.caption("Note: Rule-based tense change has limitations.")
        if not spacy_nlp:
            st.warning("spaCy model not loaded. Tense changing disabled.")
        else:
            tense_options = ["Present", "Past", "Future"]
            # Use st.session_state.selected_tense to preserve radio button choice
            idx_tense = tense_options.index(st.session_state.selected_tense) if st.session_state.selected_tense in tense_options else 0
            selected_tense = st.radio(
                "Target tense:", options=tense_options, key="tense_select_key",
                horizontal=True, index=idx_tense,
                on_change=lambda: setattr(st.session_state, 'selected_tense', st.session_state.tense_select_key)
            )
            st.session_state.selected_tense = selected_tense # Ensure it's set

            if st.button("Apply Tense Change"):
                if base_sentence_for_ops:
                    with st.spinner(f"Changing tense to {selected_tense}..."):
                        result = change_tense_rule_based(base_sentence_for_ops, spacy_nlp, selected_tense)
                    st.session_state.tense_changed_sentence = result
                    st.session_state.tone_changed_sentence = "" # Clear tone if tense is changed on base
                else:
                    st.session_state.tense_changed_sentence = "No sentence for tense change."

            if st.session_state.tense_changed_sentence:
                st.markdown(f"**Tense Changed ({selected_tense}):**")
                st.markdown(f"> {st.session_state.tense_changed_sentence}")
                if st.button("Analyze this Tense-Changed Version", key="use_tense_for_analysis"):
                    st.session_state.last_processed_for_analysis = st.session_state.tense_changed_sentence
                    if spacy_nlp:
                         with st.spinner("Generating POS tags..."):
                            st.session_state.pos_tags_df = get_pos_tags(st.session_state.tense_changed_sentence, spacy_nlp)
                    st.session_state.tense_changed_sentence = "" # Clear it as it's now the base
                    st.session_state.tone_changed_sentence = "" # Also clear tone
                    st.rerun()
        st.markdown("---") # Separator

        # --- Tone Change ---
        st.subheader("🎨 Change Tone (Simplified)")
        st.caption("Note: Tone change is rule-based and experimental, especially 'Gen Z'.")
        tone_options = ["Formal", "Friendly", "Gen Z (Experimental)"]
        idx_tone = tone_options.index(st.session_state.selected_tone) if st.session_state.selected_tone in tone_options else 0
        selected_tone = st.radio(
            "Target tone:", options=tone_options, key="tone_select_key",
            horizontal=True, index=idx_tone,
            on_change=lambda: setattr(st.session_state, 'selected_tone', st.session_state.tone_select_key)
        )
        st.session_state.selected_tone = selected_tone # Ensure it's set

        if st.button("Apply Tone Change"):
            if base_sentence_for_ops:
                with st.spinner(f"Changing tone to {selected_tone}..."):
                    result = change_tone_simplified(base_sentence_for_ops, selected_tone)
                st.session_state.tone_changed_sentence = result
                st.session_state.tense_changed_sentence = "" # Clear tense if tone is changed on base
            else:
                st.session_state.tone_changed_sentence = "No sentence for tone change."

        if st.session_state.tone_changed_sentence:
            st.markdown(f"**Tone Changed ({selected_tone}):**")
            st.markdown(f"> {st.session_state.tone_changed_sentence}")
            if st.button("Analyze this Tone-Changed Version", key="use_tone_for_analysis"):
                st.session_state.last_processed_for_analysis = st.session_state.tone_changed_sentence
                if spacy_nlp:
                     with st.spinner("Generating POS tags..."):
                        st.session_state.pos_tags_df = get_pos_tags(st.session_state.tone_changed_sentence, spacy_nlp)
                st.session_state.tone_changed_sentence = "" # Clear it as it's now the base
                st.session_state.tense_changed_sentence = "" # Also clear tense
                st.rerun()

    with col2:
        st.subheader("📊 Part-of-Speech (POS) Tags")
        if not st.session_state.pos_tags_df.empty:
            st.dataframe(st.session_state.pos_tags_df, use_container_width=True)
        elif spacy_nlp and st.session_state.last_processed_for_analysis:
             st.caption("POS tags for the current base sentence. Process or analyze a version to update.")
        elif not spacy_nlp:
            st.warning("spaCy model not available for POS tagging.")
        else:
            st.caption("Process a sentence to see POS tags.")

    st.divider()
    if st.session_state.corrected_sentence and \
       st.session_state.last_processed_for_analysis != st.session_state.corrected_sentence:
        if st.button("🔄 Reset to Analyze Corrected Version", key="reset_base_for_analysis"):
            st.session_state.last_processed_for_analysis = st.session_state.corrected_sentence
            st.session_state.tense_changed_sentence = ""
            st.session_state.tone_changed_sentence = ""
            st.session_state.selected_tense = None
            st.session_state.selected_tone = None
            if spacy_nlp:
                with st.spinner("Generating POS tags for corrected sentence..."):
                    st.session_state.pos_tags_df = get_pos_tags(st.session_state.corrected_sentence, spacy_nlp)
            st.rerun()

# Sidebar
st.sidebar.markdown("---")
st.sidebar.header("About this App")
st.sidebar.info(
    "This application uses Natural Language Processing to:\n"
    "- Correct grammatical errors.\n"
    "- Attempt to change sentence tense.\n"
    "- Attempt to change sentence tone (simplified).\n"
    "- Provide Part-of-Speech tagging."
)
st.sidebar.caption("Powered by LanguageTool, spaCy, and PyInflect.")