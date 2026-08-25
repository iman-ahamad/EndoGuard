# ============================================================
# EndoGuard Backend — Entity Consistency Framework for GI Endoscopy VQA
# Run this entire file in a single Kaggle notebook cell (GPU T4/P100)
# ============================================================

# ── 0. Install packages ──────────────────────────────────────
import subprocess, sys

def pip_install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", pkg], capture_output=True)

print("📦 Installing packages...")
pip_install("git+https://github.com/huggingface/transformers.git")
pip_install("git+https://github.com/huggingface/peft.git")
pip_install("accelerate>=1.14.0")
pip_install("bitsandbytes>=0.50.0")
pip_install("open_clip_torch")
for pkg in ["fastapi==0.115.0", "uvicorn[standard]==0.30.0", "pyngrok==7.1.6",
            "Pillow>=10.0.0", "python-multipart>=0.0.9", "nest-asyncio>=1.6.0",
            "sentencepiece>=0.2.0", "psutil", "scikit-learn", "python-dotenv>=1.0.0"]:
    pip_install(pkg)
print("✅ All packages installed")

# ── 1. Imports ────────────────────────────────────────────────
import os, json, time, re, random, io, base64, gc
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import torch
import torch.nn as nn
import psutil
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

# ── 2. Paths & Tokens ────────────────────────────────────────
# Configure these with environment variables or Kaggle Secrets. Do not commit
# tokens or machine-specific dataset paths to GitHub.
DPO_CHECKPOINT = os.getenv("DPO_CHECKPOINT", "/kaggle/input/YOUR_DATASET/checkpoints/medgemma_4b/final")
VERIFIER_CHECKPOINT = os.getenv("VERIFIER_CHECKPOINT", "/kaggle/input/YOUR_DATASET/verifier/classifier_head.pth")
SAMPLE_IMAGES_DIR = os.getenv("SAMPLE_IMAGES_DIR", "/kaggle/input/YOUR_DATASET/kvasir_vqa_final/images")
IMAGE_LABELS_PATH = os.getenv("IMAGE_LABELS_PATH", "/kaggle/input/YOUR_DATASET/kvasir_vqa_final/metadata/image_labels.json")

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    NGROK_AUTH_TOKEN = s.get_secret("NGROK_AUTH_TOKEN")
    HF_TOKEN = s.get_secret("HF_TOKEN")
except Exception:
    pass
if not NGROK_AUTH_TOKEN:
    raise ValueError("NGROK_AUTH_TOKEN required — set it in Kaggle Secrets")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN required — set it in Kaggle Secrets")

# ── 3. Load DPO fine-tuned MedGemma ──────────────────────────
print("🔄 Loading MedGemma-4B with QLoRA adapter...")
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", token=HF_TOKEN)
model = PeftModel.from_pretrained(base_model, DPO_CHECKPOINT)
model.eval()
print("✅ MedGemma + DPO adapter loaded")

# ── 4. Entity Verifier ───────────────────────────────────────
from open_clip import create_model_from_pretrained

CLASSIFIER_ENTITIES = ["polyp", "ulcerative colitis", "oesophagitis"]
FULL_ENTITY_VOCAB = [
    "polyp", "ulcerative colitis", "oesophagitis", "z-line",
    "pylorus", "cecum", "tube", "polyp snare", "biopsy forceps",
]

ENTITY_COLORS = {
    "polyp":              {"color": "#ef4444", "type": "abnormality"},
    "ulcerative colitis": {"color": "#f97316", "type": "abnormality"},
    "oesophagitis":       {"color": "#eab308", "type": "abnormality"},
    "z-line":             {"color": "#3b82f6", "type": "landmark"},
    "pylorus":            {"color": "#3b82f6", "type": "landmark"},
    "cecum":              {"color": "#3b82f6", "type": "landmark"},
    "tube":               {"color": "#22c55e", "type": "instrument"},
    "polyp snare":        {"color": "#22c55e", "type": "instrument"},
    "biopsy forceps":     {"color": "#22c55e", "type": "instrument"},
}


class EntityClassifierHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_entities=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_entities),
        )

    def forward(self, x):
        return self.classifier(x)


class EntityVerifier:
    def __init__(self, classifier_path, threshold=0.5, device="cuda"):
        self.device = device
        self.threshold = threshold
        self.biomedclip, self.preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.biomedclip = self.biomedclip.to(device).eval()
        for p in self.biomedclip.parameters():
            p.requires_grad = False

        self.head = EntityClassifierHead(input_dim=512)
        state = torch.load(classifier_path, map_location=device, weights_only=True)
        self.head.load_state_dict(state)
        self.head = self.head.to(device).eval()

    def predict_entities(self, image):
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.biomedclip.encode_image(img_tensor)
            if features.dim() > 2:
                features = features.mean(dim=1)
            logits = self.head(features)
            probs = torch.sigmoid(logits).squeeze(0)
        scores = {CLASSIFIER_ENTITIES[i]: float(probs[i]) for i in range(3)}
        predicted = {name for name, score in scores.items() if score >= self.threshold}
        return predicted, scores

    @staticmethod
    def _is_negated(entity, text):
        negation_patterns = [
            r"no\s+" + re.escape(entity),
            r"no\s+\w+\s+" + re.escape(entity),
            r"not\s+\w*\s*" + re.escape(entity),
            r"without\s+\w*\s*" + re.escape(entity),
            r"absence\s+of\s+\w*\s*" + re.escape(entity),
            re.escape(entity) + r"s?\s+(?:is|are|was|were)\s+not",
            re.escape(entity) + r"s?\s+(?:is|are|was|were)\s+not\s+(?:visible|seen|detected|identified|found|present|observed)",
        ]
        text_lower = text.lower()
        mention_pattern = r"\b" + re.escape(entity) + r"s?\b"
        mentions = list(re.finditer(mention_pattern, text_lower))
        if not mentions:
            return False
        negated_count = 0
        for mention in mentions:
            start = max(0, mention.start() - 60)
            end = min(len(text_lower), mention.end() + 60)
            context = text_lower[start:end]
            for neg_pat in negation_patterns:
                if re.search(neg_pat, context, flags=re.IGNORECASE):
                    negated_count += 1
                    break
        return negated_count == len(mentions)

    @staticmethod
    def _split_response(vlm_response):
        response = vlm_response.strip()
        header_text = ""
        answer_body = response
        header_match = re.match(
            r"[Dd]etected\s+entities\s*:\s*(.*?)(?:\.\s+|\.\n|\n)", response
        )
        if header_match:
            header_text = header_match.group(1).strip()
            answer_body = response[header_match.end():].strip()
        return header_text, answer_body

    def verify(self, image, vlm_response):
        header_text, answer_body = self._split_response(vlm_response)

        header_entities = set()
        for entity in FULL_ENTITY_VOCAB:
            pattern = r"\b" + re.escape(entity) + r"s?\b"
            if re.findall(pattern, header_text, flags=re.IGNORECASE):
                header_entities.add(entity)
        # Check aliases in header
        for alias, canonical in ENTITY_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"s?\b"
            if re.findall(pattern, header_text, flags=re.IGNORECASE):
                header_entities.add(canonical)
        # Check substring patterns in header
        header_lower = header_text.lower()
        for canonical, patterns in ENTITY_SUBSTRING_PATTERNS.items():
            if canonical not in header_entities:
                for pat in patterns:
                    if pat in header_lower:
                        header_entities.add(canonical)
                        break

        body_entities = set()
        negated_entities = set()
        for entity in FULL_ENTITY_VOCAB:
            pattern = r"\b" + re.escape(entity) + r"s?\b"
            if re.findall(pattern, answer_body, flags=re.IGNORECASE):
                if self._is_negated(entity, answer_body):
                    negated_entities.add(entity)
                else:
                    body_entities.add(entity)
        # Check aliases in body
        for alias, canonical in ENTITY_ALIASES.items():
            pattern = r"\b" + re.escape(alias) + r"s?\b"
            if re.findall(pattern, answer_body, flags=re.IGNORECASE):
                if canonical not in body_entities and canonical not in negated_entities:
                    body_entities.add(canonical)
        # Check substring patterns in body
        body_lower = answer_body.lower()
        for canonical, patterns in ENTITY_SUBSTRING_PATTERNS.items():
            if canonical not in body_entities and canonical not in negated_entities:
                for pat in patterns:
                    if pat in body_lower:
                        body_entities.add(canonical)
                        break

        vlm_entities = set(body_entities)
        for entity in header_entities:
            if entity not in negated_entities and entity not in body_entities:
                vlm_entities.add(entity)

        predicted, scores = self.predict_entities(image)

        flagged, verified, unchecked = [], [], []
        for entity in vlm_entities:
            if entity not in CLASSIFIER_ENTITIES:
                unchecked.append(entity)
            elif entity in predicted:
                verified.append(entity)
            else:
                flagged.append(entity)

        return {
            "flagged_entities": sorted(flagged),
            "verified_entities": sorted(verified),
            "unchecked_entities": sorted(unchecked),
            "negated_entities": sorted(negated_entities),
            "header_entities": sorted(header_entities),
            "body_entities": sorted(body_entities),
            "all_mentioned": sorted(vlm_entities),
            "classifier_scores": scores,
            "needs_correction": len(flagged) > 0,
        }


print("🔄 Loading EntityVerifier (BiomedCLIP + MLP)...")
verifier = EntityVerifier(VERIFIER_CHECKPOINT, threshold=0.5)
print("✅ EntityVerifier loaded")

# ── 5. Inference helpers ──────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are a medical assistant analyzing gastrointestinal endoscopy images. "
    "First list the medical entities you detect, then answer the question. "
    "Use this format: Detected entities: {entity list}. {answer sentence}"
)
MAX_NEW_TOKENS = 256
MAX_IMAGE_SIZE = 512


def run_inference(pil_image, question):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    if max(pil_image.size) > MAX_IMAGE_SIZE:
        pil_image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

    prompt_text = f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[pil_image], return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return response, elapsed


# Spelling variants → canonical entity name (exact word boundary matches)
ENTITY_ALIASES = {
    "esophagitis": "oesophagitis",
    "oesophagitits": "oesophagitis",
    "esophagitits": "oesophagitis",
    "ulcerative-colitis": "ulcerative colitis",
    "ulcerativecolitis": "ulcerative colitis",
    "polyps": "polyp",
}

# Broader substring patterns for the 3 classifier entities
# These catch adjective forms, compound terms, and common model outputs
ENTITY_SUBSTRING_PATTERNS = {
    "oesophagitis": [
        "esophagitis", "oesophagitis",               # exact terms (either spelling)
        "esophageal", "oesophageal",                  # adjective forms
        "erosive esophag", "reflux esophag",          # common medical prefixes
        "grade a esophag", "grade b esophag",         # LA classification
        "grade c esophag", "grade d esophag",
        "la grade",                                    # Los Angeles grading
        "mucosal break", "mucosal erosion",           # descriptive findings
    ],
    "polyp": [
        "polyp",                                       # catches polyp, polyps, polypoid
    ],
    "ulcerative colitis": [
        "ulcerative colitis", "ulcerative-colitis",
        "ulcerative_colitis",
    ],
}


def extract_entities_from_response(response_text):
    """Extract entity mentions from a VLM response for highlighting.
    Uses three-tier matching: canonical → aliases → substring patterns."""
    mentioned = []
    text_lower = response_text.lower()

    # Tier 1: Check canonical entities (exact word boundary)
    for entity in FULL_ENTITY_VOCAB:
        pattern = r"\b" + re.escape(entity) + r"s?\b"
        if re.findall(pattern, text_lower):
            if entity not in mentioned:
                mentioned.append(entity)

    # Tier 2: Check aliases (exact word boundary for variant spellings)
    for alias, canonical in ENTITY_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"s?\b"
        if re.findall(pattern, text_lower):
            if canonical not in mentioned:
                mentioned.append(canonical)

    # Tier 3: Substring patterns for the 3 classifier entities
    # This catches adjective forms, compound terms, and descriptive mentions
    for canonical, patterns in ENTITY_SUBSTRING_PATTERNS.items():
        if canonical not in mentioned:
            for pat in patterns:
                if pat in text_lower:
                    mentioned.append(canonical)
                    break

    # Debug: log what we found (helps diagnose detection issues)
    print(f"   [ENTITY EXTRACT] Response preview: {text_lower[:120]}...")
    print(f"   [ENTITY EXTRACT] Detected: {mentioned}")

    return mentioned


def build_entity_highlights(entities, statuses=None):
    """Build highlighted entity list with colors."""
    highlighted = []
    for ent in entities:
        info = ENTITY_COLORS.get(ent, {"color": "#9ca3af", "type": "unknown"})
        entry = {"entity": ent, "type": info["type"], "color": info["color"]}
        if statuses and ent in statuses:
            entry["status"] = statuses[ent]
        highlighted.append(entry)
    return highlighted


def build_correction_prompt(question, initial_response, flagged, verified):
    flagged_str = ", ".join(flagged)
    verified_str = ", ".join(verified) if verified else "none"
    return (
        f"Your previous answer mentioned these entities that were NOT confirmed "
        f"by the visual classifier: {flagged_str}. "
        f"The following entities WERE confirmed: {verified_str}. "
        f"Please rewrite your answer to the question without mentioning the "
        f"unconfirmed entities. Keep the same format.\n\n"
        f"Original question: {question}\n"
        f"Original answer: {initial_response}"
    )


def apply_hybrid_correction(image, question, initial_response, ver_result):
    header_text, answer_body = EntityVerifier._split_response(initial_response)

    genuinely_claimed = []
    denied_in_body = []

    for entity in ver_result["flagged_entities"]:
        if EntityVerifier._is_negated(entity, answer_body):
            denied_in_body.append(entity)
        else:
            pattern = r"\b" + re.escape(entity) + r"s?\b"
            if re.findall(pattern, answer_body, flags=re.IGNORECASE):
                genuinely_claimed.append(entity)
            else:
                denied_in_body.append(entity)

    if len(genuinely_claimed) == 0:
        # CASE A: Header-only edit
        remaining = []
        for entity in FULL_ENTITY_VOCAB:
            pattern = r"\b" + re.escape(entity) + r"s?\b"
            if re.findall(pattern, header_text, flags=re.IGNORECASE):
                if entity not in ver_result["flagged_entities"]:
                    remaining.append(entity)
        for entity in ver_result["verified_entities"]:
            if entity not in remaining:
                remaining.append(entity)
        new_header = "Detected entities: " + (", ".join(remaining) if remaining else "none")
        final_response = new_header + ". " + answer_body
        return final_response, "header_edit"
    else:
        # CASE B: Model rewrite
        correction_prompt = build_correction_prompt(
            question, initial_response, genuinely_claimed,
            ver_result["verified_entities"],
        )
        corrected, _ = run_inference(image, correction_prompt)
        return corrected, "model_rewrite"


# ── 6. Sample gallery ────────────────────────────────────────

# All possible Kaggle mount points for the dataset
_CANDIDATE_ROOTS = [
    "/kaggle/input/kvasir-vqa-v2",
    "/kaggle/input/datasets",
    "/kaggle/input",
]

# Category display config — only the 3 classifier-verifiable abnormalities
GALLERY_CATEGORIES = {
    "polyp":              {"display": "Polyp",              "type": "abnormality", "color": "#ef4444", "count": 4},
    "ulcerative-colitis": {"display": "Ulcerative Colitis", "type": "abnormality", "color": "#f97316", "count": 3},
    "oesophagitis":       {"display": "Oesophagitis",       "type": "abnormality", "color": "#eab308", "count": 3},
}


def _find_file_any(name, roots=None):
    """Search for a file under multiple root directories."""
    for root in (roots or _CANDIDATE_ROOTS):
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None


def _find_images_dir(roots=None):
    """Find the directory containing the bulk endoscopy .jpg images."""
    for root in (roots or _CANDIDATE_ROOTS):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            jpg_count = sum(1 for f in filenames if f.lower().endswith(".jpg"))
            if jpg_count > 50:
                return dirpath
    return None


def _classify_image(labels_dict):
    """Determine the category key for an image from its labels dict.
    Handles any field name/value format by searching all values."""

    # Build a flat text from ALL values in the label dict for flexible matching
    all_values = " ".join(
        str(v).strip().lower() for v in labels_dict.values() if isinstance(v, str)
    ).replace("_", " ").replace("-", " ")

    # Match against our known categories (order matters: check specific first)
    MATCH_RULES = [
        ("ulcerative-colitis", ["ulcerative colitis", "ulcerative_colitis", "ulcerative-colitis"]),
        ("oesophagitis",       ["oesophagitis", "esophagitis"]),
        ("polyp",              ["polyp"]),
        ("z-line",             ["z-line", "z line", "zline"]),
        ("pylorus",            ["pylorus"]),
        ("cecum",              ["cecum", "caecum"]),
    ]

    for cat_key, patterns in MATCH_RULES:
        for pat in patterns:
            if pat in all_values:
                return cat_key

    return None


def get_sample_gallery():
    """Build a balanced gallery of ~10 images across all categories."""
    global SAMPLE_IMAGES_DIR

    # Step 1: Find image_labels.json
    labels_path = IMAGE_LABELS_PATH
    if not os.path.exists(labels_path):
        labels_path = _find_file_any("image_labels.json")

    if not labels_path:
        for name in ["labels.json", "metadata.json", "annotations.json"]:
            labels_path = _find_file_any(name)
            if labels_path:
                break

    if not labels_path:
        print("⚠️  No labels JSON found under /kaggle/input/ — gallery disabled")
        return []

    print(f"📁 Labels file: {labels_path}")

    # Step 2: Find images directory
    if not os.path.isdir(SAMPLE_IMAGES_DIR):
        found_dir = _find_images_dir()
        if found_dir:
            SAMPLE_IMAGES_DIR = found_dir
        else:
            print("⚠️  No images directory found — gallery disabled")
            return []
    print(f"📁 Images dir:  {SAMPLE_IMAGES_DIR}")

    # Step 3: Load labels
    with open(labels_path) as f:
        all_labels = json.load(f)
    print(f"   Total labeled images: {len(all_labels)}")

    # ── DIAGNOSTIC: print first 3 entries so we can see the actual format ──
    sample_keys = list(all_labels.keys())[:3]
    for k in sample_keys:
        print(f"   SAMPLE [{k}]: {json.dumps(all_labels[k], ensure_ascii=False)[:200]}")

    # Step 4: Bucket images by category
    buckets = {cat: [] for cat in GALLERY_CATEGORIES}
    unmatched = 0

    for img_id, labels in all_labels.items():
        cat = _classify_image(labels)
        if cat and cat in buckets:
            # Check the image actually exists on disk
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join(SAMPLE_IMAGES_DIR, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    buckets[cat].append((img_id, ext, labels))
                    break
        else:
            unmatched += 1

    # Print what we found
    for cat, items in buckets.items():
        print(f"   {GALLERY_CATEGORIES[cat]['display']:>20s}: {len(items)} images available")
    if unmatched:
        print(f"   {'Unmatched':>20s}: {unmatched} images (no category matched)")

    # Step 5: Select balanced sample
    selected = []
    for cat, cfg in GALLERY_CATEGORIES.items():
        pool = buckets.get(cat, [])
        if not pool:
            continue
        n = min(cfg["count"], len(pool))
        chosen = random.sample(pool, n)
        for img_id, ext, labels in chosen:
            selected.append({
                "id": img_id,
                "ext": ext,
                "category": cfg["display"],
                "category_key": cat,
                "type": cfg["type"],
                "color": cfg["color"],
                "labels": labels,
            })

    random.shuffle(selected)
    return selected[:12]


print("🖼️  Building sample gallery...")
GALLERY = get_sample_gallery()
print(f"✅ Gallery ready — {len(GALLERY)} images across {len(set(g['category'] for g in GALLERY))} categories")

# ── 7. Evaluation metrics ────────────────────────────────────
EVAL_RESULTS = {
    "zero_shot": {
        "ECS": 0.565, "GSS": 0.248, "CHAIRs": 0.431,
        "Refusal_Accuracy": 0.119, "Entity_F1": 0.126,
        "BERTScore": 0.785, "BLEU1": 0.164,
        "polyp_hal": 450, "uc_hal": 30, "oeso_hal": 10,
    },
    "ecf_dpo": {
        "ECS": 0.675, "GSS": 0.513, "CHAIRs": 0.357,
        "Refusal_Accuracy": 0.607, "Entity_F1": 0.364,
        "BERTScore": 0.853, "BLEU1": 0.281,
        "polyp_hal": 311, "uc_hal": 106, "oeso_hal": 37,
    },
    "ecf_dpo_with_verifier": {
        "ECS": 0.664, "GSS": 0.506, "CHAIRs": 0.348,
        "Refusal_Accuracy": 0.414, "Entity_F1": 0.331,
        "BERTScore": 0.839, "BLEU1": 0.246,
        "polyp_hal": 323, "uc_hal": 68, "oeso_hal": 29,
    },
    "curriculum_stages": {
        "stage1_easy":  {"ECS": 0.587, "GSS": 0.425, "Refusal_Accuracy": 0.194, "Entity_F1": 0.204, "CHAIRs": 0.431},
        "stage2_hard":  {"ECS": 0.654, "GSS": 0.488, "Refusal_Accuracy": 0.442, "Entity_F1": 0.304, "CHAIRs": 0.354},
        "stage3_final": {"ECS": 0.675, "GSS": 0.513, "Refusal_Accuracy": 0.607, "Entity_F1": 0.364, "CHAIRs": 0.357},
    },
    "verifier_stats": {
        "classifier_macro_f1": 0.944,
        "polyp_f1": 0.935,
        "uc_f1": 0.917,
        "oesophagitis_f1": 0.979,
        "verification_precision": 94.4,
    },
}

# ── 8. FastAPI app ────────────────────────────────────────────
app = FastAPI(title="EndoGuard API", version="1.0.0")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.get("/")
def root():
    return {
        "service": "EndoGuard — Entity Consistency Framework",
        "version": "1.0.0",
        "status": "running",
        "uptime": f"{time.time() - START_TIME:.0f}s",
        "endpoints": ["/health", "/system", "/gallery", "/metrics", "/analyze"],
    }


@app.get("/health")
def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1e6, 1),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
        }
    mem = psutil.virtual_memory()
    return {
        "status": "healthy",
        "uptime": f"{time.time() - START_TIME:.0f}s",
        "gpu_available": torch.cuda.is_available(),
        **gpu_info,
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "ram_percent": mem.percent,
        "model_loaded": True,
        "verifier_loaded": True,
    }


@app.get("/system")
def system_info():
    h = health()
    return {
        **h,
        "model": "MedGemma-4B-IT + DPO LoRA",
        "quantization": "4-bit NF4",
        "adapter_size": "94.5 MB",
        "verifier": "BiomedCLIP + MLP classifier",
        "verifier_entities": CLASSIFIER_ENTITIES,
        "full_entity_vocab": FULL_ENTITY_VOCAB,
        "max_new_tokens": MAX_NEW_TOKENS,
        "gallery_size": len(GALLERY),
    }


@app.get("/gallery")
def gallery():
    items = []
    for g in GALLERY:
        items.append({
            "id": g["id"],
            "category": g["category"],
            "category_key": g["category_key"],
            "type": g["type"],
            "color": g["color"],
            "labels": g["labels"],
            "thumbnail_url": f"/gallery/{g['id']}",
        })
    return {"images": items, "total": len(items)}


@app.get("/gallery/{img_id}")
def gallery_image(img_id: str):
    # Try to find the image — check gallery first for known extension
    img_path = None
    for g in GALLERY:
        if g["id"] == img_id:
            img_path = os.path.join(SAMPLE_IMAGES_DIR, f"{img_id}{g['ext']}")
            break

    # Fallback: try common extensions
    if not img_path or not os.path.exists(img_path):
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(SAMPLE_IMAGES_DIR, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

    if not img_path or not os.path.exists(img_path):
        raise HTTPException(404, f"Image {img_id} not found")

    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((384, 384), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/gallery/{img_id}/base64")
def gallery_image_base64(img_id: str):
    """Return image as base64 data URI — avoids cross-origin img issues with ngrok."""
    img_path = None
    for g in GALLERY:
        if g["id"] == img_id:
            img_path = os.path.join(SAMPLE_IMAGES_DIR, f"{img_id}{g['ext']}")
            break
    if not img_path or not os.path.exists(img_path):
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(SAMPLE_IMAGES_DIR, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
    if not img_path or not os.path.exists(img_path):
        raise HTTPException(404, f"Image {img_id} not found")

    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"id": img_id, "data_uri": f"data:image/jpeg;base64,{b64}"}


@app.get("/metrics")
def metrics():
    return EVAL_RESULTS


@app.post("/analyze")
async def analyze(
    image: Optional[UploadFile] = File(None),
    image_id: Optional[str] = Form(None),
    mode: str = Form("dpo_only"),
    question: str = Form("What do you see in this endoscopy image?"),
):
    # ── Load image ──
    pil_image = None
    if image and image.filename:
        data = await image.read()
        pil_image = Image.open(io.BytesIO(data))
    elif image_id:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(SAMPLE_IMAGES_DIR, f"{image_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            raise HTTPException(404, f"Gallery image {image_id} not found")
        pil_image = Image.open(img_path)
    else:
        raise HTTPException(400, "Provide either an uploaded image or an image_id")

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    if max(pil_image.size) > MAX_IMAGE_SIZE:
        pil_image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)
    img_size = f"{pil_image.size[0]}x{pil_image.size[1]}"

    t_total_start = time.time()

    # ── Run inference ──
    response, inference_time = run_inference(pil_image, question)

    if mode == "dpo_only":
        mentioned = extract_entities_from_response(response)
        highlighted = build_entity_highlights(mentioned)
        total_time = time.time() - t_total_start
        return {
            "mode": "dpo_only",
            "question": question,
            "response": response,
            "inference_time": f"{total_time:.1f}s",
            "image_size": img_size,
            "entities": {
                "mentioned": mentioned,
                "highlighted": highlighted,
            },
        }

    elif mode == "dpo_with_verifier":
        initial_response = response
        ver_result = verifier.verify(pil_image, initial_response)

        final_response = initial_response
        was_corrected = False
        correction_type = None

        if ver_result["needs_correction"]:
            final_response, correction_type = apply_hybrid_correction(
                pil_image, question, initial_response, ver_result
            )
            was_corrected = True

        mentioned = extract_entities_from_response(initial_response)
        statuses = {}
        for e in ver_result.get("verified_entities", []):
            statuses[e] = "verified"
        for e in ver_result.get("flagged_entities", []):
            statuses[e] = "flagged"
        for e in ver_result.get("unchecked_entities", []):
            statuses[e] = "unchecked"

        highlighted = build_entity_highlights(mentioned, statuses)
        total_time = time.time() - t_total_start

        return {
            "mode": "dpo_with_verifier",
            "question": question,
            "initial_response": initial_response,
            "final_response": final_response,
            "was_corrected": was_corrected,
            "correction_type": correction_type,
            "inference_time": f"{total_time:.1f}s",
            "image_size": img_size,
            "entities": {
                "mentioned": mentioned,
                "flagged": ver_result["flagged_entities"],
                "verified": ver_result["verified_entities"],
                "unchecked": ver_result["unchecked_entities"],
                "negated": ver_result["negated_entities"],
                "highlighted": highlighted,
            },
            "verifier": {
                "classifier_scores": ver_result["classifier_scores"],
                "threshold": 0.5,
                "precision": f"{EVAL_RESULTS['verifier_stats']['verification_precision']}%",
            },
        }
    else:
        raise HTTPException(400, f"Unknown mode: {mode}. Use 'dpo_only' or 'dpo_with_verifier'")


# ── 9. Launch server ──────────────────────────────────────────
from pyngrok import ngrok, conf
import nest_asyncio, uvicorn

nest_asyncio.apply()

PORT = int(os.getenv("PORT", "8000"))
ENABLE_NGROK = os.getenv("ENABLE_NGROK", "true").lower() in {"1", "true", "yes"}

if ENABLE_NGROK:
    conf.get_default().auth_token = NGROK_AUTH_TOKEN
    try:
        ngrok.kill()
    except Exception:
        pass

    tunnel = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url
    print(f"\n{'='*60}")
    print("🚀 EndoGuard API is LIVE!")
    print(f"🌐 Public URL: {public_url}")
    print("📋 Set this as VITE_API_BASE_URL in frontend/.env")
    print(f"{'='*60}\n")

uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
