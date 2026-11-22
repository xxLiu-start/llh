"""Learning-only evidence adapters. Never alter evaluation reports or rewards."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from .trajectory_compactor import compact, redact


def learning_audit(text, round_index):
    from ..auditor_agent import parse_audit_report
    original = asdict(parse_audit_report(text, round_index))
    changes = []
    pattern = re.compile(r'^(\s*(?:[-*]\s*)?(?:阻断约束|blocking\s+(?:acceptance\s+)?constraints?)\s*[:：]\s*)'
                         r'(无|没有|暂无|none|nothing)\s*([（(])([^（）()\n]*)[）)]\s*[。.]?\s*$', re.I)
    def normalize(match):
        explanation = match.group(4)
        # Ambiguous statements such as "none (but X is still blocking)" are
        # never normalized into a clean assertion.
        if re.search(r'但是|但仍|不过|however|\bbut\b|\bexcept\b|仍有.*阻断|仍存在.*阻断|存在.*阻断项', explanation, re.I):
            return match.group(0)
        changes.append('NO_BLOCKING_PARENTHETICAL_FORMAT')
        return match.group(1)+match.group(2)+'。'+explanation
    adapted = '\n'.join(pattern.sub(normalize, line) for line in text.splitlines())
    parsed = asdict(parse_audit_report(adapted, round_index)) if changes else original
    return original, parsed, changes


def partial_round_evidence(root, ledger_root, completed):
    """Retain saved actions from interrupted rounds, without inventing completion."""
    refs, records, audits, trajectories = {}, [], [], {}
    containers = [ledger_root] if ledger_root != root else list(root.rglob('role_orchestration'))
    if not containers and (root/'rounds').is_dir():
        containers = [root]
    aliases = {'manager': ('orchestrator','manager'), 'executor': ('task','executor'), 'auditor': ('verifier','auditor')}
    fields = {'manager': ('plan','task_state','task_contract'), 'executor': ('output',), 'auditor': ('report',)}
    for container in containers:
        for folder in sorted((container/'rounds').glob('round_*')):
            match = re.fullmatch(r'round_(\d+)', folder.name)
            if not match or int(match[1]) in completed:
                continue
            round_id = int(match[1])
            record = {'round_index': round_id, 'complete': False, 'roles': {}}
            for role, names in aliases.items():
                metadata = {}
                for name in names:
                    p = folder/(name+'_metadata.json')
                    if p.is_file():
                        metadata = json.loads(p.read_text())
                        break
                saved = []
                for field in fields[role]:
                    candidates = ([folder/(name+'_'+field+'.txt') for name in names] +
                                  ([folder/(field+'.txt')] if field in {'task_state','task_contract'} else []))
                    path = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
                    if path is None:
                        continue
                    text = redact(path.read_text(errors='replace'))
                    ref = f'partial:round={round_id}:{role}:{field}'
                    refs[ref] = {'role': role, 'text': text, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'source_path': str(path.relative_to(root)), 'completeness': 'partial',
                                 'episode_status': metadata.get('status','unknown'), 'round_index': round_id}
                    saved.append(ref)
                    if role == 'auditor' and metadata.get('status') == 'done':
                        original, parsed, changes = learning_audit(text, round_id)
                        audits.append({'reference': ref, 'original': original, 'parsed': parsed, 'normalizations': changes})
                for name in names:
                    candidates = [folder/(name+'_raw_trajectory'+suffix) for suffix in ('.jsonl','.txt')]
                    path = next((p for p in candidates if p.is_file() and p.stat().st_size), None)
                    if path is None:
                        continue
                    summary = compact(path)
                    if not summary['tool_calls'] and not summary['errors']:
                        continue
                    ref = f'partial:round={round_id}:{role}:trajectory'
                    refs[ref] = {'role':role, 'text':redact(json.dumps(summary,ensure_ascii=False)),
                        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'source_path':str(path.relative_to(root)),
                        'completeness':'partial', 'round_index':round_id, 'episode_status':metadata.get('status','unknown')}
                    trajectories.setdefault(role, []).append({'reference':str(path.relative_to(root)), 'completeness':'partial', **summary})
                    saved.append(ref)
                    break
                if saved:
                    record['roles'][role] = {'status':metadata.get('status','unknown'), 'references':saved}
            if record['roles']:
                records.append(record)
    # Native episode streams can survive cancellation even when the round's
    # *_raw_trajectory file was never finalized. Read observed tools/errors,
    # never prompt.md (an instruction is not proof of an executed action).
    native = {'manager':('orchestrator','manager'), 'executor':('cli_task','gui_task'),
              'auditor':('cli_verifier','gui_verifier')}
    for role, names in native.items():
        for name in names:
            for episode in sorted(root.glob(f'agent/**/episodes/{name}/ep*')):
                match = re.fullmatch(r'ep(\d+)',episode.name)
                if not match or int(match[1]) <= len(completed):
                    continue
                path = episode/'claude_stream.jsonl'
                if not path.is_file() or not path.stat().st_size:
                    continue
                summary = compact(path)
                if not summary['tool_calls'] and not summary['errors']:
                    continue
                ref = f'partial:native:{name}:{episode.name}:trajectory'
                refs[ref] = {'role':role,'text':redact(json.dumps(summary,ensure_ascii=False)),
                    'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'source_path':str(path.relative_to(root)),
                    'completeness':'partial','round_index':None,'episode_status':'unfinalized_round',
                    'modality':'gui' if name.startswith('gui_') else 'cli'}
                trajectories.setdefault(role,[]).append({'reference':str(path.relative_to(root)), 'completeness':'partial', **summary})
                records.append({'round_index':None,'complete':False,'native_episode':str(episode.relative_to(root)),
                                'roles':{role:{'status':'unfinalized_round','references':[ref]}}})
    return refs, records, audits, trajectories


def eligibility(*, positive, negative_audits, refs, failed_tests, runtime_failure, agent_timeout, partial_rounds):
    own = any(source.get('role') in {'manager','executor','auditor'} for source in refs.values())
    benchmark = ['benchmark-verifier'] if failed_tests and refs.get('benchmark-verifier',{}).get('text','').strip() else []
    independent = list(dict.fromkeys([*negative_audits, *benchmark]))
    negative = bool(not runtime_failure and own and independent)
    reasons = []
    if runtime_failure: reasons.append('INFRASTRUCTURE_FAILURE')
    if not own: reasons.append('NO_SAVED_ROLE_BEHAVIOR')
    if not independent: reasons.append('NO_INDEPENDENT_NEGATIVE_EVIDENCE')
    if partial_rounds: reasons.append('PARTIAL_ROUNDS_PRESENT_NO_COMPLETION_ASSUMPTIONS')
    if agent_timeout: reasons.append('TIMEOUT_IS_NOT_ROLE_CAUSALITY')
    return {'positive':bool(positive), 'negative':negative, 'review_allowed':bool(positive or negative),
            'negative_independent_evidence':independent, 'reason_codes':reasons,
            'negative_evidence_policy':'own recorded behavior plus clean independent audit (aligned/needs_revision) or actual verifier failure; causality still requires review'}
