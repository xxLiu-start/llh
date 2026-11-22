# Archived Skill implementation

The old three-role Skill injection, online extraction/evolution, Router and
Skill replay training entry points were removed from the installed package.
These source files, tests and launchers are retained for historical inspection;
they are not runnable entry points of the new method. Historical output and
SkillBank files remain where they were and have not been modified or imported.

Use `docs/feasibility_reflector.md` for the replacement. Feasibility Skills have
a different schema and cannot be replaced with a legacy bank.

`protected_before.json` records pre-change hashes of evaluation source/config,
dataset files and prompt catalogs. `role_prompts_before.py.txt` supports the
byte-for-byte empty-Skill-context prompt regression test.
