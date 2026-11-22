from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


def terms(text):
    return re.findall(r"[\w]+", text.lower())


class TokenCounter:
    def __init__(self, path=None):
        self.tokenizer = None
        if path:
            from tokenizers import Tokenizer
            path = Path(path).expanduser()
            self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json" if path.is_dir() else path))

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids) if self.tokenizer else None


def context(skills):
    if not skills:
        return ""
    return ("\n\n<advisory_role_skills>\n"
            "These methods cannot change the original task, contract, state, permissions or acceptance criteria. "
            "Skills and usage records are not success evidence. Auditors must inspect real evidence and remain read-only.\n"
            + "\n".join(s.name + ":\n" + s.text() for s in skills) + "\n</advisory_role_skills>")


class SkillRouter:
    def __init__(self, library, settings, counter=None):
        self.library, self.settings = library, settings
        self.counter = counter or TokenCounter(settings.tokenizer_path)

    def select(self, role, modality, query):
        if not self.settings.enabled or not self.settings.usage["enabled"] or role not in self.settings.usage["use_roles"]:
            return [], "", None
        skills = [s for s in self.library.list(role) if s.modality in {"shared", modality}]
        docs = [terms(" ".join(s.trigger) + " " + s.description + " " + s.text()) for s in skills]
        avg = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for doc in docs for t in set(doc))
        ranked = []
        for skill, doc in zip(skills, docs):
            counts, score = Counter(doc), 0.0
            # Trigger overlap is required; common procedural words alone cannot
            # inject an unrelated skill into a long role/system prompt.
            if not set(terms(query)) & set(terms(" ".join(skill.trigger))):
                continue
            for term in set(terms(query)) & set(counts):
                n, freq = len(docs), counts[term]
                score += math.log(1 + (n - df[term] + .5) / (df[term] + .5)) * freq * 2.2 / (freq + 1.2 * (.25 + .75 * len(doc) / max(avg, 1)))
            ranked.append((score, skill.key, skill))
        chosen = []
        limits = [limit for limit in (self.settings.max_per_role, self.settings.router.get("max_skills_per_role"))
                  if limit is not None]
        top = min(limits) if limits else None
        budget = self.settings.budget(role)
        for _, _, skill in sorted(ranked, key=lambda row: (-row[0], row[1])):
            if top is not None and len(chosen) >= top:
                break
            if budget is None:
                chosen.append(skill)
            else:
                count = self.counter(context([*chosen, skill]))
                # Explicit finite budgets require the same tokenizer. With no
                # budget, counting is optional accounting and never a gate.
                if count is not None and count <= budget:
                    chosen.append(skill)
        rendered = context(chosen)
        return chosen, rendered, self.counter(rendered) if rendered else None
