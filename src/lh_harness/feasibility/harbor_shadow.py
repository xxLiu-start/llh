"""Optional Harbor subclass: observe original MEA, without replacing its run loop."""
from __future__ import annotations

import os

from ..original_harness_eval import OriginalHarnessHarborAgent
from .legacy_observer import LegacyContractObserver
from .model import JsonModelAgent
from .reflector import FeasibilityReflector
from .schema import ReflectorConfig


class ShadowReflectorHarborAgent(OriginalHarnessHarborAgent):
    def __init__(self, *args, feasibility_reflector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflector_config = ReflectorConfig.from_dict(feasibility_reflector or {"enabled": True, "mode": "shadow"})
        if not self.reflector_config.active or self.reflector_config.mode != "shadow":
            raise ValueError("This observational adapter requires enabled shadow mode")
        self._shadow_observer = None

    async def run(self, instruction, environment, context):
        self._shadow_task = instruction
        try:
            # Original Harbor setup, role factory, orchestrator, prompts, tools,
            # verifier and result/usage accounting all remain in the parent.
            return await super().run(instruction, environment, context)
        finally:
            if self._shadow_observer is not None:
                try:
                    self._shadow_observer.finalize()
                except Exception as exc:
                    print(f"[reflector_shadow] finalization_error={type(exc).__name__}", flush=True)

    def _make_role_agents(self, **kwargs):
        roles = super()._make_role_agents(**kwargs)
        bridge = str(self.extra_env.get("LH_HARNESS_PROXY_OPENAI_BRIDGE", os.getenv("LH_HARNESS_PROXY_OPENAI_BRIDGE", "0"))) == "1"
        model = JsonModelAgent(model=kwargs["model"], api_key=kwargs["api_key"],
            base_url=kwargs["base_url"], protocol="openai" if bridge else "anthropic")
        observer = LegacyContractObserver(roles["orchestrator"], task=self._shadow_task,
            role_dir=kwargs["output_dir"]/"cua_harness"/"role_orchestration",
            reflector=FeasibilityReflector(self.reflector_config, model))
        roles["orchestrator"] = observer
        self._shadow_observer = observer
        return roles
