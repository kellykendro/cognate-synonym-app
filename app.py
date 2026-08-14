#import modules
import csv, glob, html as html_lib, json, math, os, re, time, unicodedata
from collections import defaultdict
from threading import Lock
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd
import streamlit as st

try:
	from deep_translator import GoogleTranslator
except ImportError:
	GoogleTranslator = None

try:
	from sentence_transformers import SentenceTransformer, util
except ImportError:
	SentenceTransformer = None
	util = None

try:
	from wordfreq import word_frequency
except ImportError:
	word_frequency = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#name/define items
DEFAULT_DATA_FILE = os.environ.get(
	"ETYMOLOGY_DATA_FILE",
	os.path.join(BASE_DIR, "etymology_data_unique_cleaned.jsonl")
)

#right now, the code only runs for Spanish. Some French and German items will require a bit of finessing due to the setup of the json file (see cat/katze and tiger/tiger in German, and heart/coeur and book/livre in French)
TRANSLATOR_MAP = {
	"English": "en",
	"Spanish": "es",
	"French": "fr",
	"German": "de",
}

#if you want to add the languages back, do it here; anything beyond these languages (e.g., Dutch) would need to be added manually to this script and the json file
L2_LANGUAGES = ["Spanish"]
SUPPORTED_LANGUAGES = {"English", "Spanish", "French", "German"}

LANG_CODE_MAP = {
	"en": "English",
	"enm": "Middle English",
	"ang": "Old English",
	"ang-ang": "Old English",
	"ang-wsx": "Old English",
	"es": "Spanish",
	"osp": "Old Spanish",
	"fro": "Old French",
	"frm": "Middle French",
	"fr": "French",
	"de": "German",
	"goh": "Old High German",
	"gmh": "Middle High German",
	"la": "Latin",
	"LL.": "Late Latin",
	"ML.": "Medieval Latin",
	"VL.": "Vulgar Latin",
	"gem-pro": "Proto-Germanic",
	"gmw-pro": "Proto-West Germanic",
	"itc-pro": "Proto-Italic",
	"ine-pro": "Proto-Indo-European",
	"it": "Italian",
	"pt": "Portuguese",
	"ro": "Romanian",
	"ca": "Catalan",
	"gl": "Galician",
	"xno": "Anglo-Norman",
	"pro": "Old Provençal",
	"NL.": "New Latin",
	"la-vul": "Vulgar Latin",
	"grc": "Ancient Greek",
	"ar": "Arabic",
	"xaa": "Andalusian Arabic",
}

#this part is to standardize the json file etymology data because there's a lot of inconsistency in how entries are structured
KNOWN_LANGUAGE_LABELS = (
	"Proto-Indo-European",
	"Proto-West Germanic",
	"Proto-Germanic",
	"Proto-Italic",
	"Old High German",
	"Middle High German",
	"Old Provençal",
	"Old Spanish",
	"Old French",
	"Old English",
	"Middle English",
	"Middle French",
	"Medieval Latin",
	"Late Latin",
	"Vulgar Latin",
	"New Latin",
	"Classical Latin",
	"Ancient Greek",
	"Andalusian Arabic",
	"Anglo-Norman",
	"Portuguese",
	"Romanian",
	"Catalan",
	"Galician",
	"Spanish",
	"English",
	"French",
	"German",
	"Italian",
	"Arabic",
	"Greek",
	"Latin",
)
LANG_LABEL_PATTERN = r"(?:" + "|".join(re.escape(label) for label in KNOWN_LANGUAGE_LABELS) + r")"

ETYMON_PATTERN = r"[^\s,;:()]+"

TEXT_LINE_RE = re.compile(rf"^(?:[-*]\s*)?({LANG_LABEL_PATTERN}):?\s+(.+?)$")
FROM_WITH_LANGUAGE_RE = re.compile(
	rf"\b(?:from|borrowed from|inherited from|via|ultimately from|early borrowing from)\s+"
	rf"({LANG_LABEL_PATTERN})\s+({ETYMON_PATTERN})",
	flags=re.IGNORECASE,
)
TEMPLATE_ANCESTOR_NAMES = {"inh", "inh+", "der", "der+", "bor", "bor+"}
PLUS_COMPONENT_TEMPLATE_NAMES = {"m+", "compound+", "com+"}
DERIVATIONAL_BASE_TEMPLATE_NAMES = {"suffix", "prefix", "compound", "affix", "confix", "blend", "af"}
PLUS_BASE_TRIGGER_TEMPLATE_NAMES = TEMPLATE_ANCESTOR_NAMES | PLUS_COMPONENT_TEMPLATE_NAMES


def get_etymology_templates(entry_json):
	"""Return only valid etymology-template dictionaries.

	JSONL records may store ``etymology_templates`` as null or another
	unexpected value. Treat those cases as having no templates.
	"""
	if not isinstance(entry_json, dict):
		return []
	templates = entry_json.get("etymology_templates")
	if not isinstance(templates, list):
		return []
	return [template for template in templates if isinstance(template, dict)]

GLOSS_STOPWORDS = {
	"a", "an", "and", "also", "as", "at", "be", "by", "compare", "for", "found",
	"from", "in", "into", "it", "its", "literally", "medieval", "new", "of", "on",
	"or", "per", "said", "see", "the", "this", "to", "via", "with",
}
ETY_CHAIN_SANITIZE_RE = re.compile(r'[“"][^”"]*[”"]|\([^)]*\)')
CHAINED_ANCESTOR_RE = re.compile(
	rf"\b(?:from|borrowed from|inherited from|via|ultimately from|early borrowing from)\s+"
	rf"(?:(?P<label>{LANG_LABEL_PATTERN})\s+(?P<labeled>{ETYMON_PATTERN})|"
	rf"(?P<unlabeled>(?!{LANG_LABEL_PATTERN}\b){ETYMON_PATTERN}))"
	rf"|\bof\s+(?P<ofword>(?!{LANG_LABEL_PATTERN}\b){ETYMON_PATTERN})",
	flags=re.IGNORECASE,
)
USAGE_LIKE_PATTERNS = (
	"du cange", "classical dictionary", "vulgar tongue", "said of",
	"appears in", "found also", "found in", "also in", "per ",
)
WORD_GLOSS_RE = re.compile(
	rf'({ETYMON_PATTERN})\s*\(["“]([^"”]+)["”]\)'
)
BAD_ETYMON_WORDS = {
	"the", "a", "an", "and", "or", "of", "for", "with", "from"
}

BAD_UNLABELED_FROM_WORDS = {
	"ancient", "andalusian", "arabic", "classical", "medieval", "modern",
	"old", "middle", "late", "vulgar", "proto", "greek", "latin",
}


def is_likely_non_etymon_word(word: str) -> bool:
	raw = clean_token(word)
	norm = normalize_no_star(raw)
	if not norm:
		return True
	if norm in BAD_ETYMON_WORDS or norm in BAD_UNLABELED_FROM_WORDS:
		return True
	if raw[:1].isupper() and not raw.startswith("*"):
		return True
	return False

INFLECTIONAL_OF_CUE_RE = re.compile(
	r"\b(?:genitive|accusative|dative|ablative|nominative|oblique|plural|singular|"
	r"feminine|masculine|neuter|inflection|inflected|form|variant)\s+$",
	flags=re.IGNORECASE,
)

QUOTED_GLOSS_RE = re.compile(r'[“"]([^”"]+)[”"]')
PAREN_GLOSS_RE = re.compile(r"\([^)]*?[A-Za-z][^)]*?\)")

SEE_REF_RE = re.compile(
	rf"^\s*see(?:\s+also)?\s+(?:(?P<lang>{LANG_LABEL_PATTERN})\s+)?(?P<word>{ETYMON_PATTERN})\b",
	flags=re.IGNORECASE,
)

#looking for gloss and comparing the glosses if they appear (e.g., Latin liber is both "book" and "free")
def clean_gloss_tokens(tokens):
	cleaned = set()
	for token in tokens:
		token = (token or "").strip().lower()
		if not token or token in GLOSS_STOPWORDS:
			continue
		if any(ch.isdigit() for ch in token):
			continue
		if len(token) < 3 or len(token) > 30:
			continue
		cleaned.add(token)
	return cleaned

def choose_display_gloss(gloss_tokens):
	cleaned = clean_gloss_tokens(gloss_tokens)
	if not cleaned:
		return None
	return min(cleaned, key=lambda x: (len(x.split()), len(x), x))

def is_usage_like_gloss_text(text):
	cleaned = (text or "").strip().lower()
	if not cleaned:
		return True
	if any(pat in cleaned for pat in USAGE_LIKE_PATTERNS):
		return True
	if cleaned.count(",") >= 2 or cleaned.count(";") >= 1 or cleaned.count(":") >= 1:
		return True
	words = cleaned.split()
	if len(words) > 4:
		return True
	return False

#these thresholds can be changed using the GUI as well, but these are the defaults
DEFAULT_ORTHO_SIMILARITY_THRESHOLD = 0.35
DEFAULT_SEMANTIC_THRESHOLD = 0.20
DEFAULT_FREQUENCY_THRESHOLD = 1.0

#these thresholds are for the backend and can't be changed using the GUI
MAX_DESCENT_DEPTH = 1
MIN_STEM_PREFIX = 4

DEBUG_TIMING = True
DEBUG_COUNTS = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sbert_model = None
SBERT_LOAD_ATTEMPTED = False
SBERT_LOAD_ERROR = None
SBERT_LOCK = Lock()

@st.cache_resource(show_spinner=False)
def _load_sbert_model_cached():
	"""Load the embedding model once per Streamlit server process."""
	if SentenceTransformer is None:
		return None, "sentence-transformers is not installed"
	try:
		return SentenceTransformer("all-MiniLM-L6-v2", device="cpu"), None
	except Exception as exc:
		return None, str(exc)


def get_sbert_model():
	"""Return the cached embedding model and fall back gracefully if unavailable."""
	global sbert_model, SBERT_LOAD_ATTEMPTED, SBERT_LOAD_ERROR
	if SBERT_LOAD_ATTEMPTED:
		return sbert_model
	sbert_model, SBERT_LOAD_ERROR = _load_sbert_model_cached()
	SBERT_LOAD_ATTEMPTED = True
	if SBERT_LOAD_ERROR and sbert_model is None:
		print(f"Sentence-transformer unavailable; using string fallback: {SBERT_LOAD_ERROR}", flush=True)
	return sbert_model


WORD_TO_ENTRY = {}
WORDS_BY_LANGUAGE = defaultdict(set)
EMBEDDING_CACHE = {}
ANCESTOR_TO_ENTRIES = defaultdict(set)
GLOBAL_ANCESTORS_BY_NORM = defaultdict(list)
ALL_ANCESTOR_RECORDS = []
FREQUENCY_STATUS = {
	"ready": word_frequency is not None,
	"message": (
		"English frequency per million is estimated using wordfreq."
		if word_frequency is not None
		else "Install wordfreq to get frequency data."
	),
	"path": "wordfreq package" if word_frequency is not None else None,
}

PROGRESS_LOCK = Lock()
PROGRESS_STATE = {
	"current_word": "",
	"progress": 0,
	"detail": "Idle.",
	"running": False,
	"error": None,
}

def debug_print(*args):
	print(*args, flush=True)

#ensuring matching happens in the same language families even if the json file entries are messy
LATIN_LABELS = {"Latin", "Late Latin", "Medieval Latin", "Vulgar Latin", "New Latin"}
EXCLUDED_COGSYN_ANCESTOR_LABELS = {"Proto-Indo-European"}

ROMANCE_LABELS = {
	"French", "Old French", "Middle French",
	"Spanish", "Old Spanish",
	"Italian", "Portuguese", "Romanian", "Catalan", "Galician",
	"Anglo-Norman", "Old Provençal",
	*LATIN_LABELS,
}

GERMANIC_LABELS = {
	"English", "Old English", "Middle English",
	"German", "Old High German", "Middle High German",
	"Proto-West Germanic", "Proto-Germanic",
}

def language_family(label: str) -> str:
	label = (label or "").strip()
	if label in ROMANCE_LABELS:
		return "Romance"
	if label in GERMANIC_LABELS:
		return "Germanic"
	return label

def are_langs_compatible(label_a, label_b):
	label_a = (label_a or "").strip()
	label_b = (label_b or "").strip()

	if not label_a or not label_b:
		return False
	if label_a == label_b:
		return True
	if label_a in LATIN_LABELS and label_b in LATIN_LABELS:
		return True

	fam_a = language_family(label_a)
	fam_b = language_family(label_b)
	return fam_a == fam_b and fam_a in {"Romance", "Germanic"}

#ordering the etymology chronologically
def ancestor_recency_rank(lang_label: str) -> int:
	label = (lang_label or "").strip()

	if label.startswith("Middle "):
		return 5
	if label in {
		"English", "Spanish", "French", "German",
		"Italian", "Portuguese", "Romanian", "Catalan", "Galician"
	}:
		return 4
	if label.startswith("Old "):
		return 3
	if label in {"Vulgar Latin", "Late Latin", "Medieval Latin", "Latin", "New Latin"}:
		return 2
	if label.startswith("Proto-"):
		return 1
	return 0

def ancestor_chronology_rank(lang_label: str) -> int:
	label = (lang_label or "").strip()

	if label.startswith("Proto-"):
		return 0
	if label in {"Vulgar Latin", "Late Latin", "Medieval Latin", "Latin", "New Latin"}:
		return 1
	if label.startswith("Old "):
		return 2
	if label in {
		"English", "Spanish", "French", "German",
		"Italian", "Portuguese", "Romanian", "Catalan", "Galician"
	}:
		return 3
	if label.startswith("Middle "):
		return 4
	return 5


def fallback_surface_ancestor_match_ok(ancestor_rec: dict, surface_entry: dict, surface_norm: str) -> bool:
	if is_likely_non_etymon_word(ancestor_rec.get("word", "")):
		return False
	ancestor_lang = (ancestor_rec.get("lang_label") or "").strip()
	surface_lang = (surface_entry.get("lang") or "").strip()
	if ancestor_lang in EXCLUDED_COGSYN_ANCESTOR_LABELS:
		return False
	if not are_langs_compatible(ancestor_lang, surface_lang):
		return False

	ancestor_norm = normalize_no_star(ancestor_rec.get("word", ""))
	if not ancestor_norm or not surface_norm:
		return False
	if ancestor_norm == surface_norm:
		return True
	if len(surface_norm) >= 4 and len(ancestor_norm) >= 4:
		return normalized_levenshtein_similarity(ancestor_norm, surface_norm) >= 0.8
	return False


def confirm_shared_etymology(entry_a, entry_b):
	anc_a = entry_a.get("ancestors", [])
	anc_b = entry_b.get("ancestors", [])

	for a in anc_a:
		for b in anc_b:
			a_lang = (a.get("lang_label") or "").strip()
			b_lang = (b.get("lang_label") or "").strip()

			if is_likely_non_etymon_word(a.get("word", "")) or is_likely_non_etymon_word(b.get("word", "")):
				continue

			if a_lang in EXCLUDED_COGSYN_ANCESTOR_LABELS or b_lang in EXCLUDED_COGSYN_ANCESTOR_LABELS:
				continue
			if not are_langs_compatible(a_lang, b_lang):
				continue

			if a.get("norm_word") == b.get("norm_word"):
				if glosses_conflict(set(a.get("gloss_tokens", set())),
									set(b.get("gloss_tokens", set()))):
					continue
				return True

			a_root = canonicalize_root(a.get("word", ""))
			b_root = canonicalize_root(b.get("word", ""))
			if not a_root or not b_root:
				continue

			if a_root == b_root:
				if glosses_conflict(set(a.get("gloss_tokens", set())),
									set(b.get("gloss_tokens", set()))):
					continue
				return True

			#treat closely matching Latin-family stems as directly related for derived forms like refrigerator/refrigerador
			if (a_lang in LATIN_LABELS or b_lang in LATIN_LABELS) and has_required_form_overlap(a.get("word", ""), b.get("word", ""), min_prefix=5):
				if root_similarity(a.get("word", ""), b.get("word", "")) >= 0.78:
					return True

	a_word = entry_a.get("display_word", "")
	b_word = entry_b.get("display_word", "")
	a_norm = normalize_no_star(a_word)
	b_norm = normalize_no_star(b_word)
	
	for a in anc_a:
		if fallback_surface_ancestor_match_ok(a, entry_b, b_norm):
			return True

	for b in anc_b:
		if fallback_surface_ancestor_match_ok(b, entry_a, a_norm):
			return True

	if len(a_norm) >= 4 and len(b_norm) >= 4:
		if normalized_levenshtein_similarity(a_norm, b_norm) >= 0.83:
			return True

	return False

def best_seed_label(seed_records):
	if not seed_records:
		return None
	best = max(
		seed_records,
		key=lambda rec: (ancestor_recency_rank(rec["lang_label"]), rec["lang_label"], rec["word"])
	)
	return format_ancestor_record(best)

#addressing diacritics and non-standard characters to facilitate matching
def transliterate_ligatures(text: str) -> str:
	if not text:
		return ""
	return (
		text.replace("œ", "oe")
			.replace("Œ", "oe")
			.replace("æ", "ae")
			.replace("Æ", "ae")
			.replace("ß", "ss")
	)

def normalize(text: str) -> str:
	if not text:
		return ""
	text = transliterate_ligatures(text)
	text = text.replace("cutoff", "").replace("...", "")
	text = "".join(
		c for c in unicodedata.normalize("NFKD", text)
		if unicodedata.category(c) != "Mn"
	)
	return re.sub(r"[^a-z0-9*]", "", text.lower())

def normalize_no_star(text: str) -> str:
	return normalize(text).replace("*", "")

#this makes sure that multiword phrases are searched as a unit rather than separately
def prepare_wordfreq_query(text: str) -> str:
	if text is None:
		return ""
	query = str(text).strip()
	query = query.replace("_", " ")
	query = re.sub(r"\s+", " ", query)
	return query

def get_frequency_per_million(word: str) -> float:
	if word_frequency is None:
		return 0.0
	query = prepare_wordfreq_query(word)
	if not query:
		return 0.0
	try:
		return float(word_frequency(query, "en") * 1_000_000.0)
	except Exception:
		return 0.0

def get_frequency_slider_params(cogsyn_candidate_rows):
	max_freq = max((float(row.get("frequency_per_million", 0.0)) for row in cogsyn_candidate_rows), default=0.0)

	if max_freq <= 1.0:
		return 1.0, 0.1
	if max_freq <= 10.0:
		return round(math.ceil(max_freq * 10.0) / 10.0, 1), 0.1
	if max_freq <= 100.0:
		return float(math.ceil(max_freq)), 1.0
	if max_freq <= 1000.0:
		return float(int(math.ceil(max_freq / 10.0) * 10)), 1.0
	return float(int(math.ceil(max_freq / 50.0) * 50)), 5.0

def clean_token(text: str) -> str:
	if not text:
		return ""
	text = transliterate_ligatures(text)
	text = unicodedata.normalize("NFC", str(text).strip().strip(".,;:()[]{}"))
	text = re.sub(r"^[^\w*\-]+|[^\w*\-]+$", "", text, flags=re.UNICODE)
	return text.replace("_", "").strip()

#this helps handle case markings in Latin
def canonicalize_root(root: str) -> str:
	r = normalize_no_star(root)
	if len(r) <= 4:
		return r
	for suffix in ("enum", "enus", "ena"):
		if r.endswith(suffix) and len(r) - len(suffix) >= 3:
			return r[:-len(suffix)] + "a"
	canonical_endings = (
		"orum", "arum", "ibus", "ium", "iae", "ius",
		"ae", "is", "es", "os", "as", "us", "um", "am", "em", "im", "io"
	)
	for ending in canonical_endings:
		stem = r[:-len(ending)]
		if r.endswith(ending) and len(stem) >= 5:
			return stem
	return r
def root_similarity(a: str, b: str) -> float:
	return SequenceMatcher(None, canonicalize_root(a), canonicalize_root(b)).ratio()

#separates parts of an entry when it encounters a comma/slash/stopword
def split_gloss(gloss: str):
	parts = re.split(r"[,;/]|\bor\b|\band\b", gloss.lower())
	out = set()
	for part in parts:
		part = part.strip()
		if 2 <= len(part) <= 80:
			out.add(part)
			for w in re.findall(r"[a-z]{3,}", part):
				out.add(w)
	return out
#matches glosses
def gloss_overlap(g1, g2) -> bool:
	return bool(set(g1).intersection(set(g2)))

#finds semantic similarity between target and cognate synonym
def get_embedding(text: str):
	key = text.strip().lower()
	if key not in EMBEDDING_CACHE:
		model = get_sbert_model()
		if not model:
			EMBEDDING_CACHE[key] = None
		else:
			EMBEDDING_CACHE[key] = model.encode(text, convert_to_tensor=True)
	return EMBEDDING_CACHE[key]

def semantic_similarity_score(target_entry: dict, cogsyn_candidate_entry: dict) -> float:
	model = get_sbert_model()
	if model and util:
		return float(util.cos_sim(
			get_embedding(target_entry["display_word"]),
			get_embedding(cogsyn_candidate_entry["display_word"])
		))
	return SequenceMatcher(
		None,
		target_entry["display_word"].lower(),
		cogsyn_candidate_entry["display_word"].lower()
	).ratio()

#finds orthographic overlap between translation and cognate synonym
def levenshtein_distance(a: str, b: str) -> int:
	a = normalize_no_star(a)
	b = normalize_no_star(b)

	if a == b:
		return 0
	if not a:
		return len(b)
	if not b:
		return len(a)

	prev = list(range(len(b) + 1))
	for i, ca in enumerate(a, start=1):
		curr = [i]
		for j, cb in enumerate(b, start=1):
			insert_cost = curr[j - 1] + 1
			delete_cost = prev[j] + 1
			replace_cost = prev[j - 1] + (0 if ca == cb else 1)
			curr.append(min(insert_cost, delete_cost, replace_cost))
		prev = curr
	return prev[-1]

def normalized_levenshtein_similarity(a: str, b: str) -> float:
	a_norm = normalize_no_star(a)
	b_norm = normalize_no_star(b)
	if not a_norm and not b_norm:
		return 1.0
	denom = max(len(a_norm), len(b_norm))
	if denom == 0:
		return 0.0
	dist = levenshtein_distance(a_norm, b_norm)
	return 1.0 - (dist / denom)

def common_prefix_len(a: str, b: str) -> int:
	a = canonicalize_root(a)
	b = canonicalize_root(b)
	n = min(len(a), len(b))
	i = 0
	while i < n and a[i] == b[i]:
		i += 1
	return i

def common_prefix_len_raw(a: str, b: str) -> int:
	n = min(len(a), len(b))
	i = 0
	while i < n and a[i] == b[i]:
		i += 1
	return i

def consonant_skeleton(text: str) -> str:
	root = canonicalize_root(text)
	return re.sub(r"[aeiouy]+", "", root)

def has_required_stem_overlap(a: str, b: str, min_prefix: int = MIN_STEM_PREFIX) -> bool:
	return common_prefix_len(a, b) >= min_prefix

def has_required_form_overlap(seed_word, cogsyn_candidate_word, min_prefix=3):
	s_skel = consonant_skeleton(seed_word)
	c_skel = consonant_skeleton(cogsyn_candidate_word)
	
	if not s_skel or not c_skel:
		return False
	if len(s_skel) >= 2 and (s_skel in c_skel or c_skel in s_skel):
		return True
	effective_prefix = min(min_prefix, len(s_skel), len(c_skel))
	if effective_prefix > 0:
		return c_skel.startswith(s_skel[:effective_prefix])
		
	return False

#identifies ancestor language from list defined at the start of the code
def normalize_lang_label(label: str) -> str:
	label = (label or "").strip()
	return LANG_CODE_MAP.get(label, label if label else "Unknown")

#remove information after first sentence punctuation
def get_parseable_etymology_text(text: str) -> str:
	text = (text or "").strip()
	if not text:
		return ""

	if text.startswith("Etymology tree"):
		return text

	match = re.search(r'\.(?=[ \t]+[A-Z“"])', text)
	if match:
		return text[:match.start() + 1].strip()

	return text

#get gloss, if available


def extract_see_reference(etymology_text: str, default_lang: str = ""):
	etymology_text = get_parseable_etymology_text(etymology_text)
	if not etymology_text:
		return None
	match = SEE_REF_RE.match(etymology_text)
	if not match:
		return None
	word = clean_token(match.group("word"))
	if not word:
		return None
	lang_label = normalize_lang_label(match.group("lang") or default_lang)
	return {
		"word": word,
		"norm_word": normalize(word),
		"lang_label": lang_label,
	}


def clone_ancestor_records(records):
	cloned = []
	for rec in records or []:
		cloned.append({
			"word": rec.get("word", ""),
			"norm_word": rec.get("norm_word", normalize(rec.get("word", ""))),
			"lang_label": rec.get("lang_label", ""),
			"gloss_tokens": set(rec.get("gloss_tokens", set())),
		})
	return cloned


def resolve_see_target_entry(entry: dict, seen_keys=None):
	see_ref = entry.get("see_reference")
	if not see_ref:
		return None
	target_key = (see_ref.get("norm_word", ""), see_ref.get("lang_label", ""))
	if not target_key[0] or not target_key[1]:
		return None
	if seen_keys is None:
		seen_keys = set()
	if target_key in seen_keys:
		return None
	seen_keys.add(target_key)
	target_entry = WORD_TO_ENTRY.get(target_key)
	if not target_entry:
		return None
	nested_target = resolve_see_target_entry(target_entry, seen_keys)
	return nested_target or target_entry


def apply_see_redirects():
	for entry_key, entry in WORD_TO_ENTRY.items():
		resolved_entry = resolve_see_target_entry(entry, seen_keys={entry_key})
		if not resolved_entry:
			continue

		combined_ancestors = clone_ancestor_records(entry.get("ancestors", []))
		combined_ancestors.extend(clone_ancestor_records(resolved_entry.get("ancestors", [])))
		entry["ancestors"] = prune_ancestors(combined_ancestors)

		entry["entry_gloss_tokens"] = set(entry.get("entry_gloss_tokens", set())) | set(
			resolved_entry.get("entry_gloss_tokens", set())
		)

		if str(entry.get("etymology_text", "")).strip().lower().startswith("see"):
			entry["etymology_text"] = resolved_entry.get("etymology_text", entry.get("etymology_text", ""))

		entry["see_resolved_from"] = {
			"display_word": resolved_entry.get("display_word", ""),
			"lang": resolved_entry.get("lang", ""),
		}

def extract_word_specific_glosses(etymology_text: str):
	out = defaultdict(set)
	etymology_text = get_parseable_etymology_text(etymology_text)
	if not etymology_text:
		return out
	for word, gloss in WORD_GLOSS_RE.findall(etymology_text):
		norm_word = normalize(word)
		if norm_word:
			out[norm_word].update(clean_gloss_tokens(split_gloss(gloss)))
	return out

#pull information from quotes and parentheses
def extract_general_gloss_tokens(etymology_text: str):
	glosses = set()
	etymology_text = get_parseable_etymology_text(etymology_text)
	if not etymology_text:
		return glosses
	for match in QUOTED_GLOSS_RE.findall(etymology_text):
		cleaned = match.strip().lower()
		if is_usage_like_gloss_text(cleaned):
			continue
		glosses.update(clean_gloss_tokens(split_gloss(cleaned)))
	for match in PAREN_GLOSS_RE.findall(etymology_text):
		if '"' in match or "“" in match or "”" in match:
			continue
		cleaned = match.strip().lower()
		if is_usage_like_gloss_text(cleaned):
			continue
		glosses.update(clean_gloss_tokens(split_gloss(cleaned)))
	return glosses

#pull out exact etymological path with glosses (if present)
#only do recursive base expansion for entries that actually contain a + style etymology
def should_expand_via_plus_etymology(entry_json: dict) -> bool:
	etymology_text = str(entry_json.get("etymology_text", "")).strip()
	if "+" in etymology_text:
		return True
	for temp in get_etymology_templates(entry_json):
		name = temp.get("name") or ""
		if "+" in name or name in PLUS_BASE_TRIGGER_TEMPLATE_NAMES:
			return True
	return False

def is_affix_like_form(word: str) -> bool:
	word = clean_token(word)
	if not word:
		return True
	norm = normalize_no_star(word)
	if not norm:
		return True
	return word.startswith("-") or word.endswith("-")


def extract_recursive_base_links(entry_json: dict):
	links = []
	seen = set()

	def add_link(word: str, lang_code: str):
		word = clean_token(word)
		if not word or word == "-" or is_affix_like_form(word):
			return
		if len(normalize_no_star(word)) < 3:
			return
		lang_label = normalize_lang_label(lang_code)
		key = (normalize(word), lang_label)
		if key in seen:
			return
		seen.add(key)
		links.append({
			"word": word,
			"norm_word": normalize(word),
			"lang_label": lang_label,
		})

	for temp in get_etymology_templates(entry_json):
		name = temp.get("name") or ""
		args = temp.get("args")
		if not isinstance(args, dict):
			args = {}

		if name == "suffix":
			add_link(args.get("2", ""), args.get("1", ""))
		elif name == "prefix":
			add_link(args.get("3", ""), args.get("1", ""))
		elif name in DERIVATIONAL_BASE_TEMPLATE_NAMES:
			lang_code = args.get("1", "")
			for key in ("2", "3", "4", "5"):
				add_link(args.get(key, ""), lang_code)

	return links


def extract_ancestors_from_templates(entry_json: dict):
	out = []
	seen = set()

	def add_record(word: str, lang_code: str, gloss: str = ""):
		word = clean_token(word)
		if word == "-":
			return
		if len(normalize_no_star(word)) < 3:
			return
		lang_label = normalize_lang_label(lang_code)
		gloss_tokens = clean_gloss_tokens(split_gloss(gloss)) if gloss else set()
		rec = {
			"word": word,
			"norm_word": normalize(word),
			"lang_label": lang_label,
			"gloss_tokens": gloss_tokens,
		}
		key = (rec["norm_word"], rec["lang_label"], tuple(sorted(rec["gloss_tokens"])))
		if rec["norm_word"] and key not in seen:
			seen.add(key)
			out.append(rec)

	for temp in get_etymology_templates(entry_json):
		name = temp.get("name") or ""
		args = temp.get("args")
		if not isinstance(args, dict):
			args = {}

		if name in TEMPLATE_ANCESTOR_NAMES:
			lang_code = args.get("2", "")
			gloss = args.get("5", "") or args.get("t", "") or args.get("gloss", "")
			source_word = args.get("3", "")

			#handle inflected forms that only list components instead of full etymology
			if "+" in name:
				source_word = args.get("4", "") or source_word
			if "+" in name and source_word == "-":
				source_word = entry_json.get("word", "")

			add_record(source_word, lang_code, gloss)
			continue

		if name in PLUS_COMPONENT_TEMPLATE_NAMES:
			lang_code = args.get("1", "")
			component_glosses = {
				"2": args.get("t1", "") or args.get("gloss1", ""),
				"3": args.get("t2", "") or args.get("gloss2", ""),
				"4": args.get("t3", "") or args.get("gloss3", ""),
				"5": args.get("t4", "") or args.get("gloss4", ""),
			}
			for key in ("2", "3", "4", "5"):
				component = args.get(key, "")
				if component == "-":
					continue
				add_record(component, lang_code, component_glosses.get(key, ""))

	return out

#remove non-etymology information
def sanitize_ety_chain_text(text: str) -> str:
	return ETY_CHAIN_SANITIZE_RE.sub("", text or "")
def should_skip_chain_match(match, chain_text: str, word: str) -> bool:
	prefix = chain_text[max(0, match.start() - 45):match.start()].lower()

	if match.groupdict().get("ofword"):
		if not INFLECTIONAL_OF_CUE_RE.search(prefix):
			return True

	tail = chain_text[match.end():]
	next_word_match = re.match(r"\s+([A-Za-z][A-Za-z\-]*)", tail)
	next_word = next_word_match.group(1).lower() if next_word_match else ""
	if next_word in {"language", "speech", "origin", "word", "words"}:
		return True
	cogsyn_candidate_norm = normalize_no_star(word)
	if cogsyn_candidate_norm in {"imitative"}:
		return True
	if cogsyn_candidate_norm in BAD_ETYMON_WORDS:
		return True

	if match.groupdict().get("unlabeled"):
		if cogsyn_candidate_norm in BAD_UNLABELED_FROM_WORDS:
			return True
		if word[:1].isupper() and not word.startswith("*"):
			return True

	if "compare " in prefix:
		return True
	return False

#find ancestor forms
def extract_ancestors_from_text(etymology_text: str):
	out = []
	seen = set()
	etymology_text = get_parseable_etymology_text(etymology_text)
	if not etymology_text:
		return out
	word_specific_glosses = extract_word_specific_glosses(etymology_text)
	for raw_line in etymology_text.splitlines():
		line = raw_line.strip()
		if not line:
			continue
		m = TEXT_LINE_RE.match(line)
		if m:
			lang_label = normalize_lang_label(m.group(1).strip())
			word = clean_token(m.group(2).split()[0])
			if len(normalize_no_star(word)) < 3:
				continue
			norm_word = normalize(word)
			if norm_word in BAD_ETYMON_WORDS:
				continue
			gloss_tokens = set(word_specific_glosses.get(norm_word, set()))

			rec = {
				"word": word,
				"norm_word": norm_word,
				"lang_label": lang_label,
				"gloss_tokens": gloss_tokens,
			}
			key = (rec["norm_word"], rec["lang_label"], tuple(sorted(rec["gloss_tokens"])))
			if rec["norm_word"] and key not in seen:
				seen.add(key)
				out.append(rec)
	current_lang_label = None
	chain_text = sanitize_ety_chain_text(etymology_text)
	for match in CHAINED_ANCESTOR_RE.finditer(chain_text):
		if match.group("label") and match.group("labeled"):
			current_lang_label = normalize_lang_label(match.group("label").strip())
			lang_label = current_lang_label
			word = clean_token(match.group("labeled"))
		else:
			if not current_lang_label:
				continue
			lang_label = current_lang_label
			word = clean_token(match.group("unlabeled") or match.group("ofword") or "")
		if len(normalize_no_star(word)) < 3:
			continue
		if should_skip_chain_match(match, chain_text, word):
			continue
		norm_word = normalize(word)
		if norm_word in BAD_ETYMON_WORDS:
			continue
		gloss_tokens = set(word_specific_glosses.get(norm_word, set()))
		rec = {
			"word": word,
			"norm_word": norm_word,
			"lang_label": lang_label,
			"gloss_tokens": gloss_tokens,
		}
		key = (rec["norm_word"], rec["lang_label"], tuple(sorted(rec["gloss_tokens"])))
		if rec["norm_word"] and key not in seen:
			seen.add(key)
			out.append(rec)
	#also collect additional same-language components from patterns like “Latin ad + gratum”
	for match in re.finditer(rf'({LANG_LABEL_PATTERN})\s+({ETYMON_PATTERN})\s*\+\s*({ETYMON_PATTERN})', chain_text):
		lang_label = normalize_lang_label(match.group(1).strip())
		for word in (clean_token(match.group(2)), clean_token(match.group(3))):
			if len(normalize_no_star(word)) < 3:
				continue
			if is_affix_like_form(word):
				continue
			norm_word = normalize(word)
			if norm_word in BAD_ETYMON_WORDS:
				continue
			gloss_tokens = set(word_specific_glosses.get(norm_word, set()))
			rec = {
				"word": word,
				"norm_word": norm_word,
				"lang_label": lang_label,
				"gloss_tokens": gloss_tokens,
			}
			key = (rec["norm_word"], rec["lang_label"], tuple(sorted(rec["gloss_tokens"])))
			if rec["norm_word"] and key not in seen:
				seen.add(key)
				out.append(rec)
	return out

def expand_entry_ancestors_via_bases(entry: dict):
	expanded = [
		{
			"word": rec["word"],
			"norm_word": rec["norm_word"],
			"lang_label": rec["lang_label"],
			"gloss_tokens": set(rec.get("gloss_tokens", set())),
		}
		for rec in entry.get("ancestors", [])
	]
	seen = {(rec["norm_word"], rec["lang_label"]) for rec in expanded}

	for link in entry.get("recursive_base_links", []):
		#use exact normalized lookup here so startup does not scan the whole language lexicon for every base link
		base_key = (normalize(link["word"]), link["lang_label"])
		base_entry = WORD_TO_ENTRY.get(base_key)
		base_seen_key = (normalize(link["word"]), link["lang_label"])
		if base_seen_key not in seen:
			seen.add(base_seen_key)
			expanded.append({
				"word": link["word"],
				"norm_word": normalize(link["word"]),
				"lang_label": link["lang_label"],
				"gloss_tokens": set(),
			})
		if not base_entry:
			continue
		for rec in base_entry.get("ancestors", []):
			key = (rec["norm_word"], rec["lang_label"])
			if key in seen:
				continue
			seen.add(key)
			expanded.append({
				"word": rec["word"],
				"norm_word": rec["norm_word"],
				"lang_label": rec["lang_label"],
				"gloss_tokens": set(rec.get("gloss_tokens", set())),
			})

	entry["ancestors"] = prune_ancestors(expanded)


#if an entry has duplicated ancestors, prune them so only 1 appears
def prune_ancestors(records):
	merged = {}
	for rec in records:
		key = (rec["norm_word"], rec["lang_label"])
		if key not in merged:
			merged[key] = {
				"word": rec["word"],
				"norm_word": rec["norm_word"],
				"lang_label": rec["lang_label"],
				"gloss_tokens": set(rec.get("gloss_tokens", set())),
			}
		else:
			merged[key]["gloss_tokens"].update(rec.get("gloss_tokens", set()))
	out = list(merged.values())
	out.sort(
		key=lambda r: (
			ancestor_chronology_rank(r["lang_label"]),
			r["lang_label"],
			r["word"].lower()
		)
	)
	return out

#format entry into something readable
def format_ancestor_record(rec: dict) -> str:
	gloss = ""
	if rec.get("gloss_tokens"):
		first_gloss = sorted(rec["gloss_tokens"])[0]
		gloss = f' (“{first_gloss}”)'
	return f'{rec["lang_label"]}: {rec["word"]}{gloss}'

#only keep relevant information (including etymology tree)
def trim_etymology_text(text: str) -> str:
	if not text:
		return "No record."
	lines = text.splitlines()
	for i, line in enumerate(lines):
		if "Etymology tree" in line:
			return "\n".join(lines[:i + 1]).strip()
	cleaned = re.split(r"\bCognate with\b|\bCognates\b", text, maxsplit=1)[0].strip()
	return cleaned if cleaned else text.strip()

def format_etymology_block(entry: dict, roots_only: bool = False) -> str:
	lines = []
	if not roots_only:
		lines.append(trim_etymology_text(entry["etymology_text"]))
	if entry["ancestors"]:
		if lines:
			lines.append("")
		lines.append("Extracted roots/ancestors:")
		for rec in entry["ancestors"]:
			lines.append(f"- {format_ancestor_record(rec)}")
	elif roots_only:
		lines.append("No extracted roots/ancestors found.")
	return "\n".join(lines)

#match etymological ancestors and check semantic/orthographic similarity to find cognate synonyms
def find_best_key(word: str, lang: str):
	norm_word = normalize(word)
	exact_key = (norm_word, lang)
	if exact_key in WORD_TO_ENTRY:
		return exact_key
	best_key = None
	best_score = 0.0
	for key in WORDS_BY_LANGUAGE.get(lang, set()):
		score = SequenceMatcher(None, norm_word, key[0]).ratio()
		if score > best_score and score >= 0.88:
			best_score = score
			best_key = key
	return best_key or exact_key

def safe_translate(word: str, source_lang: str, target_lang: str):
	source_code = TRANSLATOR_MAP[source_lang]
	target_code = TRANSLATOR_MAP[target_lang]
	if GoogleTranslator is None:
		return word
	try:
		translated = GoogleTranslator(source=source_code, target=target_code).translate(word)
	except Exception:
		translated = word
	return translated or word

def glosses_conflict(seed_gloss: set, cand_gloss: set) -> bool:
	if not seed_gloss or not cand_gloss:
		return False
	return not gloss_overlap(seed_gloss, cand_gloss)

def same_or_equivalent_ancestor(seed_rec: dict, cogsyn_candidate_rec: dict) -> bool:
	seed_gloss = set(seed_rec.get("gloss_tokens", set()))
	cand_gloss = set(cogsyn_candidate_rec.get("gloss_tokens", set()))
	seed_lang = (seed_rec.get("lang_label") or "").strip()
	cand_lang = (cogsyn_candidate_rec.get("lang_label") or "").strip()
	if seed_lang in EXCLUDED_COGSYN_ANCESTOR_LABELS or cand_lang in EXCLUDED_COGSYN_ANCESTOR_LABELS:
		return False
	langs_compatible = are_langs_compatible(seed_lang, cand_lang)
	if not langs_compatible:
		return False
	if glosses_conflict(seed_gloss, cand_gloss):
		return False
	seed_word = seed_rec.get("word", "")
	cand_word = cogsyn_candidate_rec.get("word", "")
	if is_likely_non_etymon_word(seed_word) or is_likely_non_etymon_word(cand_word):
		return False
	seed_root = canonicalize_root(seed_word)
	cand_root = canonicalize_root(cand_word)
	if not seed_root or not cand_root:
		return False
	same_lang = (seed_lang == cand_lang)
	both_latin = (seed_lang in LATIN_LABELS and cand_lang in LATIN_LABELS)
	latin_family_pair = (
		language_family(seed_lang) == "Romance"
		and language_family(cand_lang) == "Romance"
		and (seed_lang in LATIN_LABELS or cand_lang in LATIN_LABELS)
	)
	if seed_rec.get("norm_word") == cogsyn_candidate_rec.get("norm_word"):
		if same_lang or both_latin:
			return True
		return bool(seed_gloss and cand_gloss and gloss_overlap(seed_gloss, cand_gloss))
	if seed_root == cand_root:
		if same_lang or both_latin:
			return True
		return bool(seed_gloss and cand_gloss and gloss_overlap(seed_gloss, cand_gloss))
	sim = root_similarity(seed_word, cand_word)
	if min(len(seed_root), len(cand_root)) < 5:
		return False
	if not (seed_gloss and cand_gloss and gloss_overlap(seed_gloss, cand_gloss)):
		return False
	if not has_required_form_overlap(seed_word, cand_word, min_prefix=3):
		return False
	if same_lang:
		return sim >= 0.88
	if both_latin or latin_family_pair:
		return sim >= 0.76
	return sim >= 0.92

def clone_ancestor_record(rec):
	return {
		"lang_label": rec.get("lang_label", ""),
		"word": rec.get("word", ""),
		"norm_word": rec.get("norm_word", normalize(rec.get("word", ""))),
		"gloss_tokens": set(rec.get("gloss_tokens", set())),
	}

def enrich_seed_ancestors(seed_records):
	enriched = {}
	for rec in seed_records:
		key = (rec.get("norm_word"), rec.get("lang_label"))
		enriched[key] = clone_ancestor_record(rec)
	for seed in seed_records:
		if not seed.get("gloss_tokens"):
			continue
		for cand in ALL_ANCESTOR_RECORDS:
			if same_or_equivalent_ancestor(seed, cand):
				key = (cand.get("norm_word"), cand.get("lang_label"))
				if key not in enriched:
					enriched[key] = clone_ancestor_record(cand)
				enriched[key]["gloss_tokens"].update(seed.get("gloss_tokens", set()))
				enriched[key]["gloss_tokens"].update(cand.get("gloss_tokens", set()))
	return list(enriched.values())

def get_matching_ancestor_words(seed_rec: dict, entry: dict):
	matched = []
	for anc in entry["ancestors"]:
		if same_or_equivalent_ancestor(seed_rec, anc):
			matched.append(anc["word"])
	if matched:
		return sorted(set(matched))
	return []

def collect_cogsyn_candidate_entries_for_seed(seed_rec: dict, target_key, translated_word, depth=1):
	results = defaultdict(set)
	for entry_key in WORDS_BY_LANGUAGE["English"]:
		entry = WORD_TO_ENTRY[entry_key]
		matched_words = get_matching_ancestor_words(seed_rec, entry)
		if matched_words:
			results[entry_key].update(matched_words)
	return results

#analysis
def analyze_word(target_word: str, l2_lang: str, progress_callback=None, direct_translation: str | None = None):
	overall_start = time.time()

	def update_progress(stage_fraction, detail):
		if progress_callback:
			progress_callback(stage_fraction, detail)
	
	#each of these update_progress lines are for the progress overlay
	update_progress(0.02, "Starting analysis")

	supplied_translation = (direct_translation or "").strip()
	translation_method = "user-supplied" if supplied_translation else "automatic"

	t0 = time.time()
	if supplied_translation:
		update_progress(0.10, "Using supplied translation")
		translated_word = supplied_translation
	else:
		update_progress(0.10, "Translating word")
		translated_word = safe_translate(target_word, "English", l2_lang)
	t1 = time.time()

	update_progress(0.20, "Matching dictionary entries")

	target_key = find_best_key(target_word, "English")
	translated_key = find_best_key(translated_word, l2_lang)

	target_entry = WORD_TO_ENTRY.get(target_key, {
		"display_word": target_word,
		"lang": "English",
		"etymology_text": "No record.",
		"ancestors": [],
		"entry_gloss_tokens": set(),
	})

	translated_entry = WORD_TO_ENTRY.get(translated_key, {
		"display_word": translated_word,
		"lang": l2_lang,
		"etymology_text": "No record.",
		"ancestors": [],
		"entry_gloss_tokens": set(),
	})

	is_direct_cognate = False
	if target_entry["display_word"].lower() == "dog" and translated_entry["display_word"].lower() == "hund":
		is_direct_cognate = False
	else:
		is_direct_cognate = confirm_shared_etymology(target_entry, translated_entry)

	empty_result = {
		"target_word": target_entry["display_word"],
		"target_language": "English",
		"target_word_etymology": format_etymology_block(target_entry, roots_only=True),
		"translated_word": translated_entry["display_word"],
		"translated_language": l2_lang,
		"translation_method": translation_method,
		"translated_word_etymology": format_etymology_block(translated_entry, roots_only=False),
		"is_direct_cognate": is_direct_cognate,
		"all_cogsyn_candidates": [],
		"synonym_groups": [],
		"frequency_slider_max": 1.0,
		"frequency_slider_step": 0.1,
	}

	if is_direct_cognate:
		update_progress(1.00, "Direct cognate detected")
		return empty_result

	translated_entry_gloss = set(translated_entry.get("entry_gloss_tokens", set()))
	target_surface_gloss = clean_gloss_tokens(split_gloss(target_entry.get("display_word", "")))

	approved_seed_ancestors = []
	for rec in translated_entry.get("ancestors", []):
		lang_label = rec.get("lang_label", "")
		if lang_label in {"Unknown", *EXCLUDED_COGSYN_ANCESTOR_LABELS}:
			continue

		rec_gloss_tokens = set(rec.get("gloss_tokens", set()))
		seed_gloss_tokens = set(rec_gloss_tokens) or set(translated_entry_gloss)

		if (
			target_surface_gloss
			and has_required_form_overlap(rec.get("word", ""), translated_entry.get("display_word", ""), min_prefix=2)
		):
			seed_gloss_tokens.update(target_surface_gloss)

		approved_seed_ancestors.append({
			"lang_label": rec.get("lang_label", ""),
			"word": rec.get("word", ""),
			"norm_word": rec.get("norm_word", normalize(rec.get("word", ""))),
			"gloss_tokens": seed_gloss_tokens,
		})

	merged_seed_ancestors = {}
	for rec in approved_seed_ancestors:
		key = (rec.get("norm_word"), rec.get("lang_label"))
		if key not in merged_seed_ancestors:
			merged_seed_ancestors[key] = {
				"lang_label": rec.get("lang_label", ""),
				"word": rec.get("word", ""),
				"norm_word": rec.get("norm_word", ""),
				"gloss_tokens": set(rec.get("gloss_tokens", set())),
			}
		else:
			merged_seed_ancestors[key]["gloss_tokens"].update(rec.get("gloss_tokens", set()))
			if not merged_seed_ancestors[key]["word"] and rec.get("word"):
				merged_seed_ancestors[key]["word"] = rec.get("word", "")

	approved_seed_ancestors = sorted(
		merged_seed_ancestors.values(),
		key=lambda r: (
			ancestor_chronology_rank(r.get("lang_label", "")),
			r.get("lang_label", ""),
			r.get("word", "").lower(),
		),
	)
	approved_seed_ancestors = enrich_seed_ancestors(approved_seed_ancestors)

	update_progress(0.35, "Preparing ancestor seeds")

	t2 = time.time()
	update_progress(0.50, "Finding possible cognate syonyms")
	raw_cogsyn_candidate_pairs = []
	for seed_rec in approved_seed_ancestors:
		cogsyn_candidate_map = collect_cogsyn_candidate_entries_for_seed(
			seed_rec,
			target_key=target_key,
			translated_word=translated_entry["display_word"],
			depth=MAX_DESCENT_DEPTH,
		)

		seed_label = format_ancestor_record(seed_rec)
		seed_word = seed_rec.get("word", "")

		if DEBUG_COUNTS:
			debug_print(
				"seed ancestor:", seed_label,
				"cogsyn_candidate keys:", len(cogsyn_candidate_map),
				"cogsyn_candidate words:", [WORD_TO_ENTRY[k]["display_word"] for k in list(cogsyn_candidate_map.keys())[:20]],
			)
		for cogsyn_candidate_key, matched_words in cogsyn_candidate_map.items():
			for matched_word in matched_words:
				raw_cogsyn_candidate_pairs.append((seed_label, seed_word, cogsyn_candidate_key, matched_word))

	t3 = time.time()

	update_progress(0.75, "Scoring possible cognate synonyms")

	cogsyn_candidate_scores = {}
	cogsyn_candidate_seed_records = defaultdict(list)

	seed_label_to_record = {}
	for seed_rec in approved_seed_ancestors:
		seed_label_to_record[format_ancestor_record(seed_rec)] = seed_rec

	for seed_label, seed_word, cogsyn_candidate_key, matched_ancestor_word in raw_cogsyn_candidate_pairs:
		cogsyn_candidate_entry = WORD_TO_ENTRY[cogsyn_candidate_key]
		cogsyn_candidate_word = cogsyn_candidate_entry["display_word"]
		ortho_trans_to_cogsyn_candidate = normalized_levenshtein_similarity(
			translated_entry["display_word"],
			cogsyn_candidate_word,
		)
		ortho_sim = ortho_trans_to_cogsyn_candidate
		sem_sim = semantic_similarity_score(target_entry, cogsyn_candidate_entry)
		rounded_sem = round(sem_sim, 3)
		rounded_ortho = round(ortho_sim, 3)
		if cogsyn_candidate_word not in cogsyn_candidate_scores:
			cogsyn_candidate_scores[cogsyn_candidate_word] = {
				"similarity": rounded_sem,
				"ortho_similarity": rounded_ortho,
			}
		else:
			if rounded_sem > cogsyn_candidate_scores[cogsyn_candidate_word]["similarity"]:
				cogsyn_candidate_scores[cogsyn_candidate_word]["similarity"] = rounded_sem
			if rounded_ortho > cogsyn_candidate_scores[cogsyn_candidate_word]["ortho_similarity"]:
				cogsyn_candidate_scores[cogsyn_candidate_word]["ortho_similarity"] = rounded_ortho
		if seed_label in seed_label_to_record:
			cogsyn_candidate_seed_records[cogsyn_candidate_word].append(seed_label_to_record[seed_label])

	t4 = time.time()

	update_progress(0.90, "Finalizing results")

	all_cogsyn_candidates = []
	grouped_items = defaultdict(list)

	for word, vals in sorted(cogsyn_candidate_scores.items(), key=lambda x: (-x[1]["similarity"], x[0].lower())):
		frequency_per_million = round(get_frequency_per_million(word), 3)
		best_label = best_seed_label(cogsyn_candidate_seed_records.get(word, []))
		
		cand = {
			"word": word,
			"similarity": vals["similarity"],
			"ortho_similarity": vals["ortho_similarity"],
			"frequency_per_million": frequency_per_million,
			"best_ancestor": best_label,
		}
		all_cogsyn_candidates.append(cand)

		if best_label:
			grouped_items[best_label].append({
				"word": word,
				"similarity": vals["similarity"],
				"ortho_similarity": vals["ortho_similarity"],
				"frequency_per_million": frequency_per_million,
			})

	synonym_groups = []
	for ancestor_label in sorted(
		grouped_items.keys(),
		key=lambda lbl: (-ancestor_recency_rank(lbl.split(":")[0]), lbl)
	):
		items = sorted(
			grouped_items[ancestor_label],
			key=lambda x: (-x["similarity"], x["word"].lower())
		)
		synonym_groups.append({
			"ancestor": ancestor_label,
			"items": items,
		})

	frequency_slider_max, frequency_slider_step = get_frequency_slider_params(all_cogsyn_candidates)
	overall_end = time.time()

	update_progress(1.00, "Finished")

	return {
		"target_word": target_entry["display_word"],
		"target_language": "English",
		"target_word_etymology": format_etymology_block(target_entry, roots_only=True),
		"translated_word": translated_entry["display_word"],
		"translated_language": l2_lang,
		"translation_method": translation_method,
		"translated_word_etymology": format_etymology_block(translated_entry, roots_only=False),
		"is_direct_cognate": False,
		"all_cogsyn_candidates": all_cogsyn_candidates,
		"synonym_groups": synonym_groups,
		"frequency_slider_max": frequency_slider_max,
		"frequency_slider_step": frequency_slider_step,
	}

#load data from entries
def build_indices(data_file: str):
	if not os.path.exists(data_file):
		raise FileNotFoundError(f"Could not find etymology data file: {data_file}")

	WORD_TO_ENTRY.clear()
	WORDS_BY_LANGUAGE.clear()
	EMBEDDING_CACHE.clear()
	ANCESTOR_TO_ENTRIES.clear()
	GLOBAL_ANCESTORS_BY_NORM.clear()
	ALL_ANCESTOR_RECORDS.clear()

	with open(data_file, "r", encoding="utf-8") as f:
		for line in f:
			try:
				data = json.loads(line)
			except Exception:
				continue
			if not isinstance(data, dict):
				continue
			word = str(data.get("word", "")).strip()
			lang = str(data.get("lang", "")).strip()
			ety = str(data.get("etymology_text", "")).strip()
			if not word or lang not in SUPPORTED_LANGUAGES:
				continue
			text_ancestors = extract_ancestors_from_text(ety)
			template_ancestors = extract_ancestors_from_templates(data)
			ancestors = prune_ancestors(text_ancestors + template_ancestors)
			recursive_base_links = extract_recursive_base_links(data) if should_expand_via_plus_etymology(data) else []
			key = (normalize(word), lang)
			entry = {
				"display_word": word,
				"lang": lang,
				"etymology_text": ety,
				"ancestors": ancestors,
				"recursive_base_links": recursive_base_links,
				"entry_gloss_tokens": extract_general_gloss_tokens(ety),
				"see_reference": extract_see_reference(ety, lang),
			}
			if key not in WORD_TO_ENTRY:
				WORD_TO_ENTRY[key] = entry
				WORDS_BY_LANGUAGE[lang].add(key)
	for entry in WORD_TO_ENTRY.values():
		if entry.get("recursive_base_links"):
			expand_entry_ancestors_via_bases(entry)

	apply_see_redirects()

	for entry_key, entry in WORD_TO_ENTRY.items():
		for rec in entry["ancestors"]:
			anc_key = (rec["norm_word"], rec["lang_label"])
			ANCESTOR_TO_ENTRIES[anc_key].add(entry_key)
			GLOBAL_ANCESTORS_BY_NORM[rec["norm_word"]].append(rec)
			ALL_ANCESTOR_RECORDS.append(rec)	

#make sure everything is ready before analysis
def ensure_resources_loaded():
	if not WORD_TO_ENTRY:
		build_indices(DEFAULT_DATA_FILE)

def reset_progress():
	with PROGRESS_LOCK:
		PROGRESS_STATE["current_word"] = ""
		PROGRESS_STATE["progress"] = 0
		PROGRESS_STATE["detail"] = "Idle."
		PROGRESS_STATE["running"] = False
		PROGRESS_STATE["error"] = None

def set_progress(current_word="", progress=0, detail="Working...", running=True, error=None):
	with PROGRESS_LOCK:
		PROGRESS_STATE["current_word"] = current_word or ""
		PROGRESS_STATE["progress"] = int(max(0, min(100, round(progress))))
		PROGRESS_STATE["detail"] = detail
		PROGRESS_STATE["running"] = running
		PROGRESS_STATE["error"] = error

def build_progress_callback(word_index: int, total_words: int, current_word: str):
	total_words = max(int(total_words or 1), 1)
	def callback(stage_fraction: float, detail: str):
		clamped_fraction = max(0.0, min(1.0, float(stage_fraction)))
		overall_progress = ((word_index - 1) + clamped_fraction) / total_words * 100.0
		set_progress(
			current_word=current_word,
			progress=overall_progress,
			detail=detail,
			running=True,
			error=None,
		)
	return callback

###streamlit changes

@st.cache_resource(show_spinner=False)
def load_index_bundle(data_file: str, data_mtime_ns: int):
    """Build the dictionary indices once and return the objects needed after Streamlit reruns."""
    build_indices(data_file)
    return {
        "WORD_TO_ENTRY": WORD_TO_ENTRY,
        "WORDS_BY_LANGUAGE": WORDS_BY_LANGUAGE,
        "EMBEDDING_CACHE": EMBEDDING_CACHE,
        "ANCESTOR_TO_ENTRIES": ANCESTOR_TO_ENTRIES,
        "GLOBAL_ANCESTORS_BY_NORM": GLOBAL_ANCESTORS_BY_NORM,
        "ALL_ANCESTOR_RECORDS": ALL_ANCESTOR_RECORDS,
    }


def attach_index_bundle(bundle):
    """Reconnect cached index objects to the globals used by the analysis functions."""
    global WORD_TO_ENTRY, WORDS_BY_LANGUAGE, EMBEDDING_CACHE
    global ANCESTOR_TO_ENTRIES, GLOBAL_ANCESTORS_BY_NORM, ALL_ANCESTOR_RECORDS
    WORD_TO_ENTRY = bundle["WORD_TO_ENTRY"]
    WORDS_BY_LANGUAGE = bundle["WORDS_BY_LANGUAGE"]
    EMBEDDING_CACHE = bundle["EMBEDDING_CACHE"]
    ANCESTOR_TO_ENTRIES = bundle["ANCESTOR_TO_ENTRIES"]
    GLOBAL_ANCESTORS_BY_NORM = bundle["GLOBAL_ANCESTORS_BY_NORM"]
    ALL_ANCESTOR_RECORDS = bundle["ALL_ANCESTOR_RECORDS"]


def decode_uploaded_words(uploaded_file):
    if uploaded_file is None:
        return []
    if not uploaded_file.name.lower().endswith(".txt"):
        raise ValueError("Upload a .txt file with one English word per line.")
    try:
        raw = uploaded_file.getvalue().decode("utf-8-sig")
    except Exception as exc:
        raise ValueError("Could not read the uploaded .txt file as UTF-8 text.") from exc
    return [line.strip() for line in raw.splitlines() if line.strip()]


def dedupe_rows(rows):
    seen = set()
    out = []
    for word, supplied_translation in rows:
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append((word, supplied_translation))
    return out


def candidate_dataframe(result, sem_threshold, ortho_threshold, frequency_threshold):
    rows = []
    for cand in result.get("all_cogsyn_candidates", []):
        passes = (
            float(cand.get("similarity", 0.0)) >= sem_threshold
            and float(cand.get("ortho_similarity", 0.0)) >= ortho_threshold
            and float(cand.get("frequency_per_million", 0.0)) >= frequency_threshold
        )
        if passes:
            rows.append({
                "Ancestor": cand.get("best_ancestor") or "Unclassified",
                "Cognate synonym": cand.get("word", ""),
                "Semantic similarity": float(cand.get("similarity", 0.0)),
                "Orthographic similarity": float(cand.get("ortho_similarity", 0.0)),
                "Frequency / million": float(cand.get("frequency_per_million", 0.0)),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Ancestor", "Cognate synonym", "Semantic similarity",
            "Orthographic similarity", "Frequency / million"
        ])
    return pd.DataFrame(rows).sort_values(
        ["Ancestor", "Semantic similarity", "Cognate synonym"],
        ascending=[True, False, True],
        kind="stable",
    )


def score_histogram_dataframe(result, metric, sem_threshold, ortho_threshold, frequency_threshold):
    values = []
    for cand in result.get("all_cogsyn_candidates", []):
        sem = float(cand.get("similarity", 0.0))
        ortho = float(cand.get("ortho_similarity", 0.0))
        freq = float(cand.get("frequency_per_million", 0.0))
        if metric == "semantic":
            include = ortho >= ortho_threshold and freq >= frequency_threshold
            value = sem
        else:
            include = sem >= sem_threshold and freq >= frequency_threshold
            value = ortho
        if include:
            values.append(max(0.0, min(1.0, value)))

    counts = [0] * 10
    for value in values:
        idx = min(int(value * 10), 9)
        counts[idx] += 1
    labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    return pd.DataFrame({"Score bin": labels, "Count": counts}).set_index("Score bin")


def _escape(value):
    return html_lib.escape(str(value or ""), quote=True)


def _multiline_html(value):
    return _escape(value).replace("\n", "<br>")


def histogram_html(result, metric, sem_threshold, ortho_threshold, frequency_threshold):
    frame = score_histogram_dataframe(
        result, metric, sem_threshold, ortho_threshold, frequency_threshold
    )
    counts = [int(v) for v in frame["Count"].tolist()]
    labels = list(frame.index)
    max_count = max(counts, default=0)
    bins = []
    for label, count in zip(labels, counts):
        height = 0 if max_count == 0 else (count / max_count * 100.0)
        bins.append(
            '<div class="vh-bin">'
            f'<div class="vh-count">{count}</div>'
            '<div class="vh-bar-wrap">'
            f'<div class="vh-bar" style="height:{height:.2f}%"></div>'
            '</div>'
            f'<div class="vh-label">{_escape(label)}</div>'
            '</div>'
        )
    title = "Semantic Distribution" if metric == "semantic" else "Orthographic Distribution"
    return (
        '<div class="histogram-panel">'
        f'<h4>{title}</h4>'
        '<div class="vertical-histogram-wrap"><div class="vertical-histogram">'
        + "".join(bins)
        + '</div></div></div>'
    )


def filtered_synonym_html(result, sem_threshold, ortho_threshold, frequency_threshold):
    if result.get("is_direct_cognate"):
        return '<span class="small-note"><strong>Cognate found - not searched</strong></span>'

    groups = []
    source_groups = result.get("synonym_groups") or []
    for group in source_groups:
        visible = []
        for item in group.get("items", []):
            sem = float(item.get("similarity", 0.0) or 0.0)
            ortho = float(item.get("ortho_similarity", 0.0) or 0.0)
            freq = float(item.get("frequency_per_million", 0.0) or 0.0)
            if sem >= sem_threshold and ortho >= ortho_threshold and freq >= frequency_threshold:
                visible.append(
                    f'{_escape(item.get("word", ""))} '
                    f'(sem {sem:.3f}, ortho {ortho:.3f}, freq {freq:.2f}/mil)'
                )
        if visible:
            groups.append(
                '<div class="synonym-row-group">'
                f'<strong>{_escape(group.get("ancestor", "Unclassified"))}</strong>: '
                + '; '.join(visible)
                + '</div>'
            )

    if groups:
        return "".join(groups)
    if source_groups or result.get("all_cogsyn_candidates"):
        return '<span class="group-none-visible">None above threshold</span>'
    return '<span class="group-none-visible">No synonyms found</span>'


def result_table_html(result, sem_threshold, ortho_threshold, frequency_threshold):
    synonym_html = filtered_synonym_html(
        result, sem_threshold, ortho_threshold, frequency_threshold
    )
    translated_language = _escape(result.get("translated_language", ""))
    return f'''
    <table class="original-result-table">
      <colgroup>
        <col class="col-target">
        <col class="col-target-ety">
        <col class="col-translation">
        <col class="col-translation-ety">
        <col class="col-synonyms">
      </colgroup>
      <thead>
        <tr>
          <th>Target (EN)</th>
          <th>Target Etymology</th>
          <th>Translation ({translated_language})</th>
          <th>Translation Etymology</th>
          <th>Cognate Synonyms</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{_escape(result.get("target_word", ""))}</td>
          <td class="ety-box small-note">{_multiline_html(result.get("target_word_etymology", ""))}</td>
          <td>{_escape(result.get("translated_word", ""))}</td>
          <td class="ety-box small-note">{_multiline_html(result.get("translated_word_etymology", ""))}</td>
          <td>{synonym_html}</td>
        </tr>
      </tbody>
    </table>
    '''


def build_csv_bytes(results, thresholds_by_result):
    fieldnames = [
        "target_word", "target_language", "target_frequency_per_million",
        "translated_word", "translated_language", "translation_method",
        "translation_frequency_per_million", "is_direct_cognate",
        "target_etymology", "translation_etymology", "candidate",
        "semantic_similarity", "orthographic_similarity",
        "candidate_frequency_per_million", "best_ancestor",
        "passes_semantic_threshold", "passes_orthographic_threshold",
        "passes_frequency_threshold", "passes_all_thresholds",
    ]
    output = []
    for idx, result in enumerate(results):
        sem_t, ortho_t, freq_t = thresholds_by_result.get(
            idx,
            (DEFAULT_SEMANTIC_THRESHOLD, DEFAULT_ORTHO_SIMILARITY_THRESHOLD, 0.0),
        )
        target_freq = round(get_frequency_per_million(result.get("target_word", "")), 3)
        translation_freq = round(get_frequency_per_million(result.get("translated_word", "")), 3)
        candidates = result.get("all_cogsyn_candidates", [])
        if not candidates:
            candidates = [{}]
        for cand in candidates:
            sem = float(cand.get("similarity", 0.0) or 0.0)
            ortho = float(cand.get("ortho_similarity", 0.0) or 0.0)
            freq = float(cand.get("frequency_per_million", 0.0) or 0.0)
            output.append({
                "target_word": result.get("target_word", ""),
                "target_language": result.get("target_language", "English"),
                "target_frequency_per_million": target_freq,
                "translated_word": result.get("translated_word", ""),
                "translated_language": result.get("translated_language", ""),
                "translation_method": result.get("translation_method", ""),
                "translation_frequency_per_million": translation_freq,
                "is_direct_cognate": bool(result.get("is_direct_cognate")),
                "target_etymology": result.get("target_word_etymology", ""),
                "translation_etymology": result.get("translated_word_etymology", ""),
                "candidate": cand.get("word", ""),
                "semantic_similarity": sem if cand else "",
                "orthographic_similarity": ortho if cand else "",
                "candidate_frequency_per_million": freq if cand else "",
                "best_ancestor": cand.get("best_ancestor", ""),
                "passes_semantic_threshold": (sem >= sem_t) if cand else "",
                "passes_orthographic_threshold": (ortho >= ortho_t) if cand else "",
                "passes_frequency_threshold": (freq >= freq_t) if cand else "",
                "passes_all_thresholds": (
                    sem >= sem_t and ortho >= ortho_t and freq >= freq_t
                ) if cand else "",
            })

    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output)
    return buffer.getvalue().encode("utf-8-sig")


APP_CSS = r'''
<style>
:root { color-scheme: light; }
html, body, [class*="css"] { font-family: Arial, sans-serif; }
[data-testid="stAppViewContainer"] { background: #fff; color: #111; }
[data-testid="stHeader"], [data-testid="stToolbar"], footer { visibility: hidden; height: 0; }
[data-testid="stMainBlockContainer"] { max-width: none; padding: 24px 24px 60px 24px; line-height: 1.4; }
h1, h2, h3, h4 { margin-bottom: 0.4em !important; color: #111; }
h1 { font-size: 2em !important; font-weight: 700 !important; }
h2 { font-size: 1.5em !important; font-weight: 700 !important; }
h4 { font-size: 1em !important; font-weight: 700 !important; }
.st-key-search_form { padding: 16px; border: 1px solid #ccc; border-radius: 8px; margin-bottom: 28px; background: #fff; }
.st-key-search_form [data-testid="stWidgetLabel"] p { font-weight: 600; color: #111; }
.st-key-l2_language, .st-key-initial_semantic, .st-key-initial_ortho, .st-key-initial_frequency, .st-key-word_input, .st-key-direct_translation { max-width: 300px; }
.st-key-file_input { max-width: 520px; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { border-radius: 0; min-height: 34px; }
[data-baseweb="select"] > div { border-radius: 0 !important; min-height: 34px; }
[data-testid="stFileUploaderDropzone"] { padding: 8px 10px; min-height: auto; }
[data-testid="stFileUploaderDropzone"] small { display: none; }
[data-testid="stFileUploader"] section { border-radius: 4px; }
.st-key-run_analysis button { padding: 8px 14px; border: 1px solid #777; border-radius: 2px; background: #f3f3f3; color: #111; font-weight: 400; width: auto; min-height: 36px; }
.st-key-run_analysis button:hover { border-color: #555; background: #e9e9e9; color: #111; }
.help-text { font-size: 0.95em; color: #444; margin-top: -6px; margin-bottom: 12px; }
.small-note { font-size: 0.92em; color: #555; }
.warning { color: #7a5200; background: #fff4d6; border: 1px solid #e0c36d; padding: 10px 12px; border-radius: 6px; margin: 12px 0; }
.error-box { color: #b00020; font-weight: 700; margin: 12px 0; }
.download-panel { border: 1px solid #ccc; border-radius: 8px; padding: 12px 14px; margin: 18px 0 8px 0; background: #fafafa; }
.st-key-download_panel { border: 1px solid #ccc; border-radius: 8px; padding: 12px 14px; margin: 18px 0 8px 0; background: #fafafa; }
.st-key-csv_download button { padding: 8px 14px; border: 1px solid #777; border-radius: 6px; background: #fff; color: #111; font-weight: 700; width: auto; }
.st-key-csv_download button:hover { background: #f0f0f0; color: #111; border-color: #777; }
.result-header { margin-top: 30px; padding-top: 12px; border-top: 2px solid #ddd; }
.result-header h2 { margin: 0 0 0.4em 0; }
.frequency-summary { margin: 6px 0 12px 0; color: #333; }
.direct-cognate-alert { background: #e8f5e9; border: 1px solid #2e7d32; color: #2e7d32; padding: 10px; border-radius: 4px; margin-bottom: 10px; font-weight: bold; }
[class*="st-key-sliders_panel_"] { border: 1px solid #ccc; border-radius: 8px; padding: 14px 16px 2px 16px; background: #fff; min-height: 100%; }
.threshold-label { font-weight: 600; margin-top: 1px; }
.threshold-display { font-weight: 400; margin-left: 5px; }
.histogram-panel { border: 1px solid #ccc; border-radius: 8px; padding: 14px 16px; background: #fff; }
.histogram-panel h4 { margin: 0 0 0.4em 0; }
.vertical-histogram-wrap { width: 100%; overflow-x: auto; padding-top: 8px; }
.vertical-histogram { height: 260px; min-width: 420px; display: flex; align-items: end; gap: 10px; padding: 12px 8px 28px 8px; border-left: 1px solid #bbb; border-bottom: 1px solid #bbb; box-sizing: border-box; }
.vh-bin { flex: 1 1 0; min-width: 30px; display: flex; flex-direction: column; align-items: center; justify-content: end; gap: 6px; height: 100%; }
.vh-count { font-size: 0.85em; color: #333; min-height: 1em; }
.vh-bar-wrap { width: 100%; height: 180px; display: flex; align-items: end; justify-content: center; }
.vh-bar { width: 70%; min-width: 8px; max-width: 42px; background: #4a78b0; border-radius: 4px 4px 0 0; }
.vh-label { font-size: 0.78em; text-align: center; line-height: 1.1; word-break: break-word; }
.original-result-table { border-collapse: collapse; width: 100%; margin-top: 16px; margin-bottom: 24px; table-layout: fixed; }
.original-result-table th, .original-result-table td { border: 1px solid #ccc; padding: 10px; vertical-align: top; text-align: left; }
.original-result-table th { background: #f5f5f5; font-weight: 700; }
.original-result-table .col-target { width: 90px; }
.original-result-table .col-target-ety { width: 220px; }
.original-result-table .col-translation { width: 170px; }
.original-result-table .col-translation-ety { width: 38%; }
.original-result-table .col-synonyms { width: calc(62% - 480px); }
.ety-box { white-space: normal; word-break: break-word; overflow-wrap: anywhere; }
.synonym-row-group { margin-bottom: 5px; }
.group-none-visible { color: #555; font-style: italic; }
.progress-host { display: flex; justify-content: center; margin: 20px 0; }
.loading-box { background: #fff; border: 1px solid #ccc; border-radius: 10px; padding: 22px 26px; box-shadow: 0 6px 18px rgba(0,0,0,0.12); text-align: center; min-width: 340px; max-width: 420px; }
.progress-word { font-size: 1.1em; font-weight: 700; margin-top: 8px; margin-bottom: 4px; word-break: break-word; }
.progress-detail { font-size: 0.92em; color: #555; margin-bottom: 8px; }
.progress-percent { font-size: 0.95em; font-weight: 700; margin-top: 8px; }
.progress-shell { width: 100%; height: 14px; margin-top: 14px; background: #e6e6e6; border: 1px solid #cfcfcf; border-radius: 999px; overflow: hidden; }
.progress-bar { height: 100%; background: #4a78b0; border-radius: 999px; }
@media (max-width: 1100px) { .original-result-table { table-layout: auto; display: block; overflow-x: auto; } }
</style>
'''


def progress_box_html(word, detail, percent):
    pct = max(0, min(100, int(percent)))
    return f'''
    <div class="progress-host">
      <div class="loading-box">
        <div><strong>Processing…</strong></div>
        <div class="small-note">Please wait while the analysis runs.</div>
        <div class="progress-word">{_escape(word or "Preparing…")}</div>
        <div class="progress-detail">{_escape(detail or "Starting analysis")}</div>
        <div class="progress-shell"><div class="progress-bar" style="width:{pct}%"></div></div>
        <div class="progress-percent">{pct}%</div>
      </div>
    </div>
    '''


def run_analysis(rows, l2_selected, frequency_threshold):
    results = []
    total_words = len(rows)
    progress_placeholder = st.empty()
    progress_placeholder.markdown(progress_box_html("Preparing…", "Starting analysis", 0), unsafe_allow_html=True)
    for idx, (word, supplied_translation) in enumerate(rows, start=1):
        def callback(stage_fraction, detail, *, _idx=idx, _word=word):
            clamped = max(0.0, min(1.0, float(stage_fraction)))
            overall = ((_idx - 1) + clamped) / total_words
            progress_placeholder.markdown(progress_box_html(_word, detail, int(round(overall * 100))), unsafe_allow_html=True)
        result = analyze_word(word, l2_selected, progress_callback=callback, direct_translation=supplied_translation or None)
        result["initial_frequency_threshold"] = min(frequency_threshold, result.get("frequency_slider_max", 1.0))
        results.append(result)
    progress_placeholder.markdown(progress_box_html(rows[-1][0], "Analysis complete", 100), unsafe_allow_html=True)
    time.sleep(0.15)
    progress_placeholder.empty()
    return results


def render_result(result, result_index, run_id, initial_sem, initial_ortho, initial_freq):
    target_freq = get_frequency_per_million(result.get("target_word", ""))
    translation_freq = get_frequency_per_million(result.get("translated_word", ""))
    st.markdown(f'''<div class="result-header"><h2>Result {result_index + 1}: {_escape(result.get("target_word", ""))}</h2><div class="small-note frequency-summary">Target frequency: {target_freq:.2f}/mil; Translation frequency: {translation_freq:.2f}/mil</div></div>''', unsafe_allow_html=True)
    if result.get("is_direct_cognate"):
        st.markdown(f'''<div class="direct-cognate-alert">Direct Cognate Detected: “{_escape(result.get("target_word", ""))}” and “{_escape(result.get("translated_word", ""))}” share a direct etymological root. No further analysis has been carried out.</div>''', unsafe_allow_html=True)
        sem_threshold, ortho_threshold, frequency_threshold = initial_sem, initial_ortho, initial_freq
    else:
        controls, charts = st.columns([0.72, 1.75], gap="large")
        with controls:
            with st.container(border=True, key=f"sliders_panel_{run_id}_{result_index}"):
                sem_threshold = st.slider("Visible semantic threshold", 0.0, 1.0, float(initial_sem), 0.01, key=f"sem_{run_id}_{result_index}")
                ortho_threshold = st.slider("Visible orthographic threshold", 0.0, 1.0, float(initial_ortho), 0.01, key=f"ortho_{run_id}_{result_index}")
                freq_max = max(float(result.get("frequency_slider_max", 1.0)), float(initial_freq), 0.1)
                freq_step = max(float(result.get("frequency_slider_step", 0.1)), 0.1)
                frequency_threshold = st.slider("Visible wordfreq estimated frequency threshold (per million)", 0.0, freq_max, min(float(initial_freq), freq_max), freq_step, key=f"freq_{run_id}_{result_index}")
        with charts:
            chart_left, chart_right = st.columns(2, gap="medium")
            with chart_left:
                st.markdown(histogram_html(result, "semantic", sem_threshold, ortho_threshold, frequency_threshold), unsafe_allow_html=True)
            with chart_right:
                st.markdown(histogram_html(result, "orthographic", sem_threshold, ortho_threshold, frequency_threshold), unsafe_allow_html=True)
    st.markdown(result_table_html(result, sem_threshold, ortho_threshold, frequency_threshold), unsafe_allow_html=True)
    return sem_threshold, ortho_threshold, frequency_threshold


def main():
    st.set_page_config(page_title="Cognate Synonym Evaluation", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown("<h1>Cognate Synonym Evaluation</h1>", unsafe_allow_html=True)
    data_path = Path(DEFAULT_DATA_FILE)
    if not data_path.exists():
        st.markdown(f'<div class="error-box">Could not find the etymology data file: {_escape(data_path)}. Place etymology_data_unique_cleaned.jsonl beside app.py or set ETYMOLOGY_DATA_FILE.</div>', unsafe_allow_html=True)
        st.stop()
    default_frequency = DEFAULT_FREQUENCY_THRESHOLD if FREQUENCY_STATUS["ready"] else 0.0
    with st.container(key="search_form"):
        l2_selected = st.selectbox("L2 language", L2_LANGUAGES, index=0, key="l2_language")
        semantic_threshold = st.number_input("Initial semantic similarity threshold (0 to 1)", 0.0, 1.0, float(DEFAULT_SEMANTIC_THRESHOLD), 0.01, key="initial_semantic")
        st.markdown('<div class="help-text">Visible semantic filter. Can be adjusted after analysis.</div>', unsafe_allow_html=True)
        ortho_threshold = st.number_input("Initial orthographic similarity threshold (0 to 1)", 0.0, 1.0, float(DEFAULT_ORTHO_SIMILARITY_THRESHOLD), 0.01, key="initial_ortho")
        st.markdown('<div class="help-text">Visible orthographic filter. Can be adjusted after analysis.</div>', unsafe_allow_html=True)
        frequency_threshold = st.number_input("Initial wordfreq estimated frequency threshold (per million)", min_value=0.0, value=float(default_frequency), step=0.1, key="initial_frequency")
        st.markdown('<div class="help-text">Visible wordfreq estimated frequency filter. Can be adjusted after analysis.</div>', unsafe_allow_html=True)
        word_input = st.text_input("Single English target word", placeholder="e.g., book", key="word_input")
        batch_selected = st.session_state.get("file_input") is not None
        direct_translation = st.text_input(f"Direct {l2_selected} translation (optional)", placeholder="Leave blank to translate automatically (Google Translate)", disabled=batch_selected, key="direct_translation")
        if batch_selected:
            direct_translation = ""
        st.markdown('<div class="help-text">Available only for one manually entered word. Batch TXT uploads always use automatic translation.</div>', unsafe_allow_html=True)
        st.markdown('<div class="help-text" style="margin-top:4px">Or upload a .txt file with one English target word per line. Batch words are translated automatically.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload .txt file", type=["txt"], key="file_input")
        run_clicked = st.button("Run analysis", key="run_analysis")
    if FREQUENCY_STATUS["ready"]:
        source = f' Source: {FREQUENCY_STATUS["path"]}' if FREQUENCY_STATUS.get("path") else ""
        st.markdown(f'<div class="small-note">{_escape(FREQUENCY_STATUS["message"])}{_escape(source)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warning">{_escape(FREQUENCY_STATUS["message"])}</div>', unsafe_allow_html=True)
    if run_clicked:
        error = None
        rows = []
        try:
            batch_words = decode_uploaded_words(uploaded)
        except ValueError as exc:
            batch_words = []
            error = str(exc)
        if direct_translation and uploaded is not None:
            error = "A direct translation can only be used for one manually entered word."
        elif direct_translation and not word_input.strip():
            error = "Enter the English word that matches the direct translation."
        if not error:
            rows.extend((word, "") for word in batch_words)
            if word_input.strip():
                rows.append((word_input.strip(), direct_translation.strip()))
            rows = dedupe_rows(rows)
            if not rows:
                error = "Please provide a word or upload a .txt file with one word per line."
        if error:
            st.markdown(f'<div class="error-box">{_escape(error)}</div>', unsafe_allow_html=True)
        else:
            try:
                with st.spinner("Loading etymology data…"):
                    bundle = load_index_bundle(
                        str(data_path),
                        data_path.stat().st_mtime_ns,
                    )
                    attach_index_bundle(bundle)

                results = run_analysis(rows, l2_selected, float(frequency_threshold))
                st.session_state["analysis_results"] = results
                st.session_state["analysis_run_id"] = int(st.session_state.get("analysis_run_id", 0)) + 1
                st.session_state["analysis_initial_sem"] = float(semantic_threshold)
                st.session_state["analysis_initial_ortho"] = float(ortho_threshold)
                st.session_state["analysis_initial_freq"] = float(frequency_threshold)
            except Exception as exc:
                st.markdown(f'<div class="error-box">Processing error: {_escape(exc)}</div>', unsafe_allow_html=True)
                with st.expander("Technical details"):
                    st.exception(exc)
    results = st.session_state.get("analysis_results")
    if not results:
        return
    run_id = st.session_state.get("analysis_run_id", 0)
    initial_sem = st.session_state.get("analysis_initial_sem", DEFAULT_SEMANTIC_THRESHOLD)
    initial_ortho = st.session_state.get("analysis_initial_ortho", DEFAULT_ORTHO_SIMILARITY_THRESHOLD)
    initial_freq = st.session_state.get("analysis_initial_freq", DEFAULT_FREQUENCY_THRESHOLD if FREQUENCY_STATUS["ready"] else 0.0)
    download_slot = st.empty()
    thresholds_by_result = {}
    for idx, result in enumerate(results):
        result_initial_freq = min(float(result.get("initial_frequency_threshold", initial_freq)), max(float(result.get("frequency_slider_max", 1.0)), 0.1))
        thresholds_by_result[idx] = render_result(result, idx, run_id, float(initial_sem), float(initial_ortho), result_initial_freq)
    csv_bytes = build_csv_bytes(results, thresholds_by_result)
    with download_slot.container():
        with st.container(border=True, key="download_panel"):
            st.download_button("Download CSV results", data=csv_bytes, file_name="cognate_synonym_results.csv", mime="text/csv", key="csv_download")
            st.markdown('<div class="small-note">The CSV contains one row per cognate synonym candidate, plus target, translation, target frequency, translation frequency, etymology, score, candidate frequency, ancestor, and threshold-pass columns.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()