import json
import os
from importlib import import_module

import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST
from . import embedding_search as es

TOP_K = 5
MIN_SCORE = 0.30      # tune this on your own test complaints
MAX_CHARS = 2000

_engine = {}          # model + section data, loaded once on the first request


def _get_engine():
    if not _engine:
        base = str(settings.BASE_DIR)
        es.DATA_FILE = os.path.join(base, "bns_sections.json")
        es.CACHE_FILE = os.path.join(base, "section_embeddings.npz")
        model = import_module("sentence_transformers").SentenceTransformer(es.MODEL_NAME)
        sections = es.load_sections()
        _engine.update(
            model=model,
            sections=sections,
            emb=es.load_or_build_embeddings(model, sections),
        )
    return _engine


@ensure_csrf_cookie
def home(request):
    return render(request, "index.html")


@require_POST
def analyze(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    complaint = str(payload.get("complaint", "")).strip()[:MAX_CHARS]
    if not complaint:
        return JsonResponse({"error": "Complaint text is required"}, status=400)

    eng = _get_engine()
    model, sections, emb = eng["model"], eng["sections"], eng["emb"]

    scores = emb @ model.encode(complaint, normalize_embeddings=True)
    results = []
    for i in np.argsort(scores)[::-1][:TOP_K]:
        score = float(scores[i])
        if score < MIN_SCORE:
            continue
        s = sections[int(i)]
        key_words = [w for w, drop in es.explain(model, emb[i], complaint) if drop > 0]
        results.append({
            "section": s.get("section", ""),
            "title": s.get("title", ""),
            "explanation": s.get("description", ""),
            "punishment": s.get("punishment", ""),
            "score": round(score, 3),
            "key_words": key_words,
        })
    return JsonResponse({"results": results})