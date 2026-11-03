"""Harbor adapter for SkillBank training on Terminal-Bench Pro.

The Terminal-Bench 2.1 evaluation adapter supplies Harbor's container and
Claude Code bridges. This adapter runs the current ``lh_harness.manager``
implementation; the training and test configs select the OpenAI tool bridge.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from harbor.models.agent.context import AgentContext
from harbor.environments.base import BaseEnvironment
from harbor_agent import CuaHarnessClaudeCodeAgent
from harbor_backends import HarborEnvironmentAdapter

from .harbor_url import claude_code_upstream_base_url
from .manager import run
from .types import EpisodeBudget, HarnessConfig


class _RoleBridge:
    """Supply the older Harbor episode adapter's turn budget and signature."""

    def __init__(self, adapter: Any, max_turns: int) -> None:
        self.adapter = adapter
        self.max_turns = max_turns

    async def run_episode(
        self,
        prompt: str,
        env: HarborEnvironmentAdapter,
        budget: EpisodeBudget,
        live_trajectory_path: str | None = None,
    ) -> Any:
        del live_trajectory_path  # The Harbor adapter saves its own episode log.
        harbor_budget = SimpleNamespace(
            max_turns=self.max_turns,
            max_duration_seconds=budget.max_duration_seconds,
        )
        return await self.adapter.run_episode(prompt, env, harbor_budget)


class SkillLearningHarborAgent(CuaHarnessClaudeCodeAgent):
    """Train the main Skill-enabled MEA loop inside a Harbor task container."""

    def __init__(
        self,
        *args: Any,
        skill_bank_path: str,
        skill_token_budget: int = 2000,
        skill_reward_reference_tokens: int = 100_000,
        role_history_chars: int = 100_000,
        role_verified_context_chars: int = 60_000,
        task_token_soft_limit: int | None = None,
        compact_runtime_context: bool = False,
        prompt_language: str = "en",
        learning_enabled: bool = True,
        candidate_path: str | None = None,
        offline_claude_binary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.skill_bank_path = str(Path(skill_bank_path).expanduser().resolve())
        self.skill_token_budget = int(skill_token_budget)
        self.skill_reward_reference_tokens = int(skill_reward_reference_tokens)
        self.role_history_chars = int(role_history_chars)
        self.role_verified_context_chars = int(role_verified_context_chars)
        if self.role_history_chars < 1 or self.role_verified_context_chars < 1:
            raise ValueError("Role context character limits must be positive")
        self.task_token_soft_limit = int(task_token_soft_limit) if task_token_soft_limit is not None else None
        self.compact_runtime_context = bool(compact_runtime_context)
        if self.task_token_soft_limit is not None and self.task_token_soft_limit < 1:
            raise ValueError("task_token_soft_limit must be positive")
        self.learning_enabled = bool(learning_enabled)
        self.candidate_path = candidate_path
        self.offline_claude_binary = bool(offline_claude_binary)
        if prompt_language not in {"en", "zh"}:
            raise ValueError("Skill prompt language must be en or zh")
        self.prompt_language = prompt_language

    @staticmethod
    def name() -> str:
        return "lh-harness-skill-learning"

    def _resolve_base_url(self) -> str | None:
        return claude_code_upstream_base_url(super()._resolve_base_url())

    async def setup(self, environment: BaseEnvironment) -> None:
        if self.offline_claude_binary:
            # Harbor skips its online installer when the mounted binary has the
            # right version, including the installer's procps step. Claude Code
            # still needs ps/pgrep for process cleanup on Python base images.
            check = await environment.exec(
                command="command -v ps >/dev/null 2>&1 && command -v pgrep >/dev/null 2>&1",
                user="root", timeout_sec=30,
            )
            if check.return_code != 0:
                install = await environment.exec(
                    command="apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y procps",
                    user="root", timeout_sec=600,
                )
                if install.return_code != 0:
                    raise RuntimeError(f"could not install procps for offline Claude Code: {install.stdout} {install.stderr}")
        await super().setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        output_dir = self.logs_dir / "skill_harness"
        output_dir.mkdir(parents=True, exist_ok=True)
        workspace_path, _ = await self._resolve_workspace_path(environment)
        env_adapter = HarborEnvironmentAdapter(environment, workspace_path=workspace_path)
        model = self._resolve_model()
        verifier_model = self._resolve_verifier_model(model)
        role_agents = self._make_role_agents(
            environment=environment,
            env_adapter=env_adapter,
            output_dir=output_dir,
            model=model,
            verifier_model=verifier_model,
            api_key=self._resolve_api_key(),
            base_url=self._resolve_base_url(),
            workspace_path=workspace_path,
            event_logger=None,
        )
        manager_agent = _RoleBridge(role_agents["orchestrator"], self._orchestrator_max_turns)
        executor_agent = _RoleBridge(role_agents["cli_task"], self._max_turns)
        auditor_agent = _RoleBridge(role_agents["cli_verifier"], self._verifier_max_turns)
        config = HarnessConfig(
            max_total_episodes=self._max_rounds,
            manager_budget=EpisodeBudget(max_duration_seconds=self._orchestrator_timeout_sec),
            cli_executor_budget=EpisodeBudget(max_duration_seconds=self._cli_task_timeout_sec),
            gui_executor_budget=EpisodeBudget(max_duration_seconds=self._cli_task_timeout_sec),
            auditor_budget=EpisodeBudget(max_duration_seconds=self._verifier_timeout_sec),
            workspace_path=workspace_path,
            harness_dir=f"{workspace_path.rstrip('/')}/.harness_skill",
            log_dir=str(output_dir / "lh_harness"),
            prompt_language=self.prompt_language,
            skill_bank_path=self.skill_bank_path,
            skill_token_budget=self.skill_token_budget,
            skill_reward_reference_tokens=self.skill_reward_reference_tokens,
            role_history_chars=self.role_history_chars,
            role_verified_context_chars=self.role_verified_context_chars,
            task_token_soft_limit=self.task_token_soft_limit,
            compact_runtime_context=self.compact_runtime_context,
            skill_update_enabled=self.learning_enabled,
            skill_benchmark_id="terminal_bench_2_1",
            skill_interaction_mode="cli",
            skill_dataset_id="terminal_bench_pro" if self.learning_enabled else "terminal_bench_2_1",
            unattended_benchmark=True,
        )
        started = time.monotonic()
        candidate_skill = None
        candidate_router_prompt = None
        if self.candidate_path:
            proposal = json.loads(Path(self.candidate_path).read_text(encoding="utf-8"))
            if proposal.get("kind") == "skill":
                candidate_skill = proposal["candidate"]
                if (not isinstance(candidate_skill, (dict, list))
                        or isinstance(candidate_skill, list) and (
                            not candidate_skill or any(not isinstance(item, dict) for item in candidate_skill)
                        )):
                    raise ValueError("Skill candidate must be a Skill or a nonempty list of Skills")
            elif proposal.get("kind") == "router":
                candidate_router_prompt = proposal["candidate"]
            else:
                raise ValueError("candidate_path must contain a Skill or Router proposal")
        report = await run(
            task=instruction,
            env=env_adapter,
            config=config,
            manager_agent=manager_agent,
            cli_executor_agent=executor_agent,
            gui_executor_agent=executor_agent,
            cli_auditor_agent=auditor_agent,
            gui_auditor_agent=auditor_agent,
            auditor_format_repair_agent=auditor_agent,
            final_response_agent=manager_agent,
            skill_candidate=candidate_skill,
            skill_router_prompt_override=candidate_router_prompt,
        )
        usage = {
            key: sum(int(adapter.usage_totals.get(key) or 0) for adapter in role_agents.values())
            for key in ("n_input_tokens", "n_output_tokens", "n_cache_tokens")
        }
        usage_partial = any(bool(getattr(adapter, "usage_partial", False)) for adapter in role_agents.values())
        context.n_input_tokens = usage["n_input_tokens"]
        context.n_output_tokens = usage["n_output_tokens"]
        context.n_cache_tokens = usage["n_cache_tokens"]
        context.metadata = {
            "variant": "lh_harness_skill_learning",
            "dataset_split": "train" if self.learning_enabled else "test",
            "skill_bank_path": self.skill_bank_path,
            "model": model,
            "verifier_model": verifier_model,
            "workspace_path": workspace_path,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completion_satisfied": bool(report.get("completion_satisfied")),
            "status": report.get("status"),
            "skill_learning": report.get("skill_learning"),
            "usage_totals": usage,
            "usage_partial": usage_partial,
            "report_path": str(output_dir / "lh_harness" / "report.json"),
        }
        (output_dir / "training_report.json").write_text(
            json.dumps({"report": report, "usage_totals": usage, "usage_partial": usage_partial}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
