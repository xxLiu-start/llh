"""Transport compatibility and skill adapters; no frozen MEA source is edited."""
from pathlib import Path
from lh_harness.original_harness_eval import OriginalHarnessHarborAgent
from lh_harness.skills.integration import SkillSession


class PristineOriginalAgent(OriginalHarnessHarborAgent):
    # Official d53df0aa assigns extra_env in its constructor; newer Harbor
    # exposes a read-only property. This compatibility setter stays outside eval.
    @property
    def extra_env(self):
        return dict(self._extra_env)

    @extra_env.setter
    def extra_env(self, value):
        self._extra_env = dict(value or {})


class SkillEnhancedAgent(PristineOriginalAgent):
    def __init__(self, *args, skills_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_session = SkillSession(skills_config or {}, Path(self.logs_dir) / "skill_evolution")

    def _make_episode_adapter(self, **kwargs):
        adapter = super()._make_episode_adapter(**kwargs)
        return self.skill_session.wrap(kwargs["role"], adapter)
