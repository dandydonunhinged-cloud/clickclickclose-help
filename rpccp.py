"""
RPCCP — Recursive Precision Cognitive Calibration Protocol
==========================================================
The microscalpel. Domain-agnostic recursive cognition engine.

Unlike RAC (blunt object, 5-axis explosion), RPCCP is surgical:
each pass doesn't just refine the answer — it evolves the question.

Type-1 recursion: Solution_n+1 = optimize(Solution_n, same_objective)
Type-2 recursion: Objective_n+1 = expand(Objective_n, new_dimensions)
                  Solution_n+1 = optimize(new_objective=Objective_n+1)

RPCCP achieves Type-2 recursion by automating the pressure that
makes thinking visible. It doesn't find answers. It finds questions.

Architecture:
  Pass 1: Solve naively (anchor bait — exposes assumptions)
  Pass 2: Self-critique (why did you choose this? what's weak?)
  Pass 3: Constraint relaxation (ignore all constraints — what's optimal?)
  Pass 4: Dimension expansion (what dimensions aren't you considering?)
  Pass 5: Paradigm shift (what if the frame itself is wrong?)
  Pass 6+: Don's push (why did you constrain your search? go outside)
  Collision: Cross-apply pass outputs — emergence detection

Each pass can run in a separate VM with a separate model.
No shared context between passes = no anchor bias.

Usage:
    engine = RPCCP()
    result = engine.run("How do we cure cancer?")

    # Or with explicit model per pass:
    engine = RPCCP(models={
        1: OllamaModel("qwen2.5:32b"),
        2: OllamaModel("hercules"),
        3: OllamaModel("phoenix"),
        4: OllamaModel("aaron"),
        5: OllamaModel("nemotron-3-super"),
    })

    # Or with VM isolation:
    engine = RPCCP(isolation="vm", vm_dir="D:/VMs/rpccp")

Authors: Don Brown & Claude (Spock) — Session 30, 2026-03-13
Framework: Don Brown, developed across sessions 1-30
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows so arrows and em-dashes don't crash cp1252
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ADAPTERS
# ============================================================================

class ModelAdapter:
    """Base class for LLM adapters. Subclass for each backend."""

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class OllamaModel(ModelAdapter):
    """Local Ollama model (localhost or Ollama Pro cloud)."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str, system: str = "") -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 4096},
        }
        if system:
            payload["system"] = system
        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=600)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            logger.error(f"[Ollama:{self.model}] {e}")
            return f"[ERROR: {e}]"

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"


class OllamaProModel(ModelAdapter):
    """Ollama Pro cloud model (api.ollama.com). Fast inference, large models."""

    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.api_key = api_key or self._load_key()

    def _load_key(self) -> str:
        import os
        # Try env var first, then .env file
        key = os.environ.get("OLLAMA_CLOUD_KEYS", "")
        if not key:
            env_path = "C:/DandyDon/.env"
            try:
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("OLLAMA_CLOUD_KEYS="):
                            key = line.split("=", 1)[1].strip()
                            break
            except FileNotFoundError:
                pass
        return key

    def generate(self, prompt: str, system: str = "") -> str:
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            r = requests.post(
                "https://ollama.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 16384,
                },
                timeout=600,
                allow_redirects=False,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[OllamaPro:{self.model}] {e}")
            return f"[ERROR: {e}]"

    @property
    def name(self) -> str:
        return f"ollama-pro:{self.model}"


class OpenRouterModel(ModelAdapter):
    """OpenRouter API for cloud models (free tier: Qwen3 235B, DeepSeek R1)."""

    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.api_key = api_key or self._load_key()

    def _load_key(self) -> str:
        import os
        return os.environ.get("OPENROUTER_API_KEY", "")

    def generate(self, prompt: str, system: str = "") -> str:
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": 0.7},
                timeout=300,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[OpenRouter:{self.model}] {e}")
            return f"[ERROR: {e}]"

    @property
    def name(self) -> str:
        return f"openrouter:{self.model}"


class CallableModel(ModelAdapter):
    """Wrap any callable(prompt, system) -> str."""

    def __init__(self, fn: Callable, label: str = "callable"):
        self._fn = fn
        self._label = label

    def generate(self, prompt: str, system: str = "") -> str:
        try:
            return self._fn(prompt, system)
        except TypeError:
            return self._fn(prompt)

    @property
    def name(self) -> str:
        return self._label


# ============================================================================
# PASS RESULTS
# ============================================================================

@dataclass
class PassResult:
    """Output of a single RPCCP pass."""
    pass_number: int
    pass_type: str  # naive, critique, unconstrained, expand, paradigm, push
    prompt_sent: str
    response: str
    model_used: str
    objective_function: str  # what "optimal" means at this pass
    dimensions_considered: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_seconds: float = 0.0


@dataclass
class CollisionResult:
    """Output of cross-pass collision analysis."""
    emergent_insights: List[Dict[str, Any]]
    real_question: str  # the question RPCCP actually found
    type2_detected: bool  # did the objective function evolve?
    objective_evolution: List[str]  # trace of how "optimal" changed


@dataclass
class RPCCPResult:
    """Complete RPCCP run output."""
    query: str
    passes: List[PassResult]
    collision: Optional[CollisionResult]
    final_synthesis: str
    total_duration: float
    models_used: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# THE PROTOCOL
# ============================================================================

# System prompts that automate the pressure Don provides manually.
# Each pass has a different cognitive posture.

PASS_SYSTEMS = {
    1: """You are Pass 1 of RPCCP — the naive solver.
Your job is to produce the BEST solution you can given the query.
Do not hedge. Do not qualify. Give your strongest answer.
This answer will be challenged. That's the point. Give it your best shot.
State your objective function explicitly: what are you optimizing for?
List every dimension you're considering.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (description/analysis of the problem) or a SOLUTION (something someone can DO)?"
If it was an answer, not a solution: state what the solution would be, then ask "why not just do that?"
and revise your output to be the actionable solution instead.""",

    2: """You are Pass 2 of RPCCP — the self-critic.
You will receive a query AND a previous answer from Pass 1.
Your job is NOT to improve the answer. Your job is to ATTACK it.
- Why did Pass 1 choose this approach?
- What assumptions did it make without justifying?
- What's the weakest link in the reasoning?
- What would an adversary say?
- What dimensions are MISSING from the analysis?
Be ruthless. The goal is to expose blind spots, not to be polite.
State what "optimal" should ACTUALLY mean, if Pass 1 got it wrong.

ANCHOR REQUIREMENT: Before concluding, you MUST:
1. Restate the ORIGINAL BASE QUESTION in your own words.
2. Explain specifically how your critique helps answer that base question better.
3. State why your reframing is superior to the initial question for achieving the stated goal.
Do NOT drift. Every insight must trace back to the base question.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (critique/analysis) or a SOLUTION (something someone can DO)?"
If it was an answer, not a solution: state what the solution would be, then ask "why not just do that?"
and revise your output to include the actionable solution.""",

    3: """You are Pass 3 of RPCCP — the unconstrained explorer.
You will receive the original query, Pass 1's answer, and Pass 2's critique.
IGNORE ALL CONSTRAINTS. Budget, feasibility, politics, timeline — throw them out.
If you could wave a magic wand, what is the TRULY optimal solution?
This is not fantasy — it's constraint relaxation to find the real optimum.
Often the unconstrained answer reveals that a constraint was artificial.
State your new objective function. What are you optimizing for now?

ANCHOR REQUIREMENT: Before concluding, you MUST:
1. Restate the ORIGINAL BASE QUESTION in your own words.
2. Explain specifically how your unconstrained answer addresses that base question better than Passes 1-2.
3. State why your reframing is superior to the initial question for achieving the stated goal.
Do NOT drift into abstractions. Every insight must connect to a concrete answer for the base question.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (description/abstraction) or a SOLUTION (something someone can DO)?"
If it was an answer, not a solution: state what the solution would be, then ask "why not just do that?"
and revise your output to be the actionable solution instead.""",

    4: """You are Pass 4 of RPCCP — the dimension expander.
You will receive everything from Passes 1-3.
Your job: what dimensions are ALL THREE passes blind to?
Think about:
- Who maintains this? Who pays for this? Who is affected but not mentioned?
- What's the organizational, political, emotional dimension?
- What would change if you added TIME as a dimension? SCALE? CULTURE?
- What adjacent domains solved a similar problem differently?
Do NOT improve the solution. EXPAND what "optimal" means.
Add at least 3 dimensions none of the previous passes considered.

ANCHOR REQUIREMENT: Before concluding, you MUST:
1. Restate the ORIGINAL BASE QUESTION in your own words.
2. For EACH new dimension you add, explain how it helps answer the base question.
3. State why expanding these dimensions brings us closer to the stated goal than the original framing.
Do NOT add dimensions that don't connect back to the base question. Expansion without relevance is noise.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (new dimensions to think about) or a SOLUTION (what to DO with those dimensions)?"
If it was an answer, not a solution: for each dimension, state the specific action it implies,
then ask "why not just do that?" and include those actions in your output.""",

    5: """You are Pass 5 of RPCCP — the paradigm breaker.
You will receive everything from Passes 1-4.
Your job is the hardest: what if the FRAME ITSELF is wrong?
- What if we're solving the wrong problem?
- What if the question assumes something false?
- What technology, method, or perspective CHANGES THE GAME entirely?
- What would a 10-year-old ask that an expert wouldn't?
This is where Type-2 recursion happens. The objective function itself must evolve.
Don't refine. REDEFINE. What is the REAL question behind the question?

ANCHOR REQUIREMENT: Before concluding, you MUST:
1. Restate the ORIGINAL BASE QUESTION in your own words.
2. Explain exactly how your paradigm shift produces a BETTER ANSWER to the base question — not just a different frame, but a more actionable one.
3. State why your new question is superior to the initial question for achieving the stated goal.
If your paradigm shift doesn't improve the answer to the base question, it's not a breakthrough — it's a tangent.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (a new way to think about the problem) or a SOLUTION (a new thing to DO)?"
If it was an answer, not a solution: state what action the paradigm shift implies,
then ask "why not just do that?" and make the action the centerpiece of your output.""",

    6: """You are Pass 6 of RPCCP — Don's push.
You will receive everything from Passes 1-5.
Pass 5 probably found something interesting but STILL constrained its search.
Your job: "Why did you constrain your search?"
- Go OUTSIDE the domain. What field, discipline, or tradition solved this differently?
- What do aviation, nursing, nuclear power, or military ops know about this?
- What ancient wisdom, indigenous knowledge, or folk practice is relevant?
- What would happen if you applied this answer to a COMPLETELY DIFFERENT domain?
Break the walls. The insight is outside the room you're searching in.

ANCHOR REQUIREMENT: Before concluding, you MUST:
1. Restate the ORIGINAL BASE QUESTION in your own words.
2. For EACH cross-domain insight, explain specifically how it applies to the base question and what concrete action it suggests.
3. State why this cross-domain perspective produces a better answer than staying inside the original domain.
Cross-domain analogies without concrete application are decoration, not insight.

SOLUTION GATE — Before concluding, you MUST ask yourself:
"Did I produce an ANSWER (interesting analogies) or a SOLUTION (what to DO based on those analogies)?"
If it was an answer, not a solution: for each analogy, state the specific intervention it suggests,
then ask "why not just do that?" and make those interventions your output.""",

    7: """You are Pass 7 of RPCCP — the collision engine.
You will receive compressed outputs from ALL previous passes.
Your job is COLLISION — cross-apply insights from different passes.
Look for:
- Insights from Pass 3 (unconstrained) that solve problems raised in Pass 2 (critique)
- Dimensions from Pass 4 that reframe Pass 5's paradigm shift
- External knowledge from Pass 6 that validates or invalidates Pass 5
- EMERGENCE: insights that exist in NO single pass but appear when passes collide

CRITICAL — ANCHOR TO THE BASE QUESTION:
Every pass was required to explain how their work addresses the original base question.
Your collision MUST honor that anchor. The final synthesis must be a BETTER ANSWER
to the base question — not a meta-commentary about how thinking works.
If the collision produces philosophy instead of actionable answers, it failed.

The output must include:
1. The REAL QUESTION (what RPCCP actually found — must be more actionable than the original)
2. Whether the objective function evolved (Type-2 recursion detected?)
3. The trace of how "optimal" changed across passes
4. The synthesized answer that no single pass could have produced
5. CONCRETE NEXT STEPS — what specifically should be done based on this synthesis

SOLUTION GATE — The collision MUST produce a SOLUTION, not an answer.
If your synthesis describes how to think about the problem, you failed.
It must describe what to DO about the problem.
Ask yourself: "Is this something a researcher could take into a lab on Monday?"
If not, ask "why not just do that?" and revise until it is.""",
}


class RPCCP:
    """
    Recursive Precision Cognitive Calibration Protocol.

    The microscalpel. Automates the cognitive pressure that forces
    Type-2 recursion: objective function evolution, not just solution refinement.

    Args:
        models: Dict mapping pass number -> ModelAdapter. If not provided,
                uses a default Ollama model for all passes.
        default_model: ModelAdapter to use for passes without explicit model.
        max_passes: Maximum passes before collision (default 6).
        isolation: "none" (shared context), "prompt" (fresh context per pass),
                   or "vm" (actual VM isolation — future).
        output_dir: Where to save run results. Default: D:/DandyDon/rpccp_runs/
        verbose: Print pass results to stdout.
    """

    def __init__(
        self,
        models: Optional[Dict[int, ModelAdapter]] = None,
        default_model: Optional[ModelAdapter] = None,
        max_passes: int = 6,
        isolation: str = "prompt",
        output_dir: str = "D:/DandyDon/rpccp_runs",
        verbose: bool = True,
    ):
        self.models = models or {}
        self.default_model = default_model or OllamaModel("qwen2.5:32b-instruct-q4_K_M")
        self.max_passes = min(max_passes, 6)  # 6 passes + collision = 7 total
        self.isolation = isolation
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def _get_model(self, pass_number: int) -> ModelAdapter:
        """Get the model for a specific pass."""
        return self.models.get(pass_number, self.default_model)

    def _build_pass_prompt(self, pass_number: int, query: str, history: List[PassResult]) -> str:
        """Build the user prompt for a pass, including relevant history."""
        parts = [f"ORIGINAL QUERY: {query}\n"]

        if pass_number == 1:
            # Pass 1 gets only the query
            pass
        elif pass_number == 2:
            # Pass 2 gets Pass 1's answer
            p1 = history[0]
            parts.append(f"PASS 1 ANSWER (objective: {p1.objective_function}):\n{p1.response}\n")
        elif pass_number <= 6:
            # Passes 3-6 get compressed history of all prior passes
            for p in history:
                # Compress each pass to key insight (first 500 chars)
                compressed = p.response[:500]
                if len(p.response) > 500:
                    compressed += "..."
                parts.append(
                    f"PASS {p.pass_number} ({p.pass_type}, objective: {p.objective_function}):\n"
                    f"{compressed}\n"
                )

        parts.append("YOUR TASK: Follow your system prompt. Be specific. Be ruthless.")
        parts.append("\nFormat your response as:")
        parts.append("OBJECTIVE FUNCTION: [what you're optimizing for]")
        parts.append("DIMENSIONS CONSIDERED: [comma-separated list]")
        parts.append("ANALYSIS: [your full response]")

        return "\n".join(parts)

    def _build_collision_prompt(self, query: str, history: List[PassResult]) -> str:
        """Build the collision prompt from all pass results."""
        parts = [f"ORIGINAL QUERY: {query}\n"]
        parts.append("=== ALL PASS RESULTS ===\n")

        for p in history:
            parts.append(f"--- PASS {p.pass_number}: {p.pass_type.upper()} ---")
            parts.append(f"Model: {p.model_used}")
            parts.append(f"Objective: {p.objective_function}")
            parts.append(f"Dimensions: {', '.join(p.dimensions_considered)}")
            parts.append(f"Response:\n{p.response}\n")

        return "\n".join(parts)

    def _parse_response(self, raw: str) -> tuple:
        """Extract objective function and dimensions from formatted response."""
        objective = ""
        dimensions = []
        analysis = raw

        lines = raw.split("\n")
        analysis_lines = []
        in_analysis = False

        for line in lines:
            if line.strip().startswith("OBJECTIVE FUNCTION:"):
                objective = line.split(":", 1)[1].strip()
            elif line.strip().startswith("DIMENSIONS CONSIDERED:"):
                dims_str = line.split(":", 1)[1].strip()
                dimensions = [d.strip() for d in dims_str.split(",") if d.strip()]
            elif line.strip().startswith("ANALYSIS:"):
                in_analysis = True
                rest = line.split(":", 1)[1].strip()
                if rest:
                    analysis_lines.append(rest)
            elif in_analysis:
                analysis_lines.append(line)

        if analysis_lines:
            analysis = "\n".join(analysis_lines)

        return objective, dimensions, analysis

    def _parse_collision(self, raw: str, history: List[PassResult]) -> CollisionResult:
        """Parse collision engine output."""
        # Try to extract structured data, fall back to raw
        objectives = [p.objective_function for p in history if p.objective_function]

        # Check if objective evolved
        unique_objectives = list(dict.fromkeys(objectives))  # preserve order, deduplicate
        type2 = len(unique_objectives) > 1

        # Try to find "REAL QUESTION" in the output
        real_question = ""
        for line in raw.split("\n"):
            lower = line.lower().strip()
            if "real question" in lower or "actual question" in lower:
                real_question = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                break

        if not real_question:
            # Take the last paragraph as the synthesis
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            real_question = paragraphs[-1] if paragraphs else raw[:200]

        return CollisionResult(
            emergent_insights=[{"raw_collision": raw}],
            real_question=real_question,
            type2_detected=type2,
            objective_evolution=unique_objectives,
        )

    def run(self, query: str, extra_passes: Optional[List[str]] = None) -> RPCCPResult:
        """
        Execute the full RPCCP protocol on a query.

        Args:
            query: The question to investigate.
            extra_passes: Additional custom pass prompts beyond the standard 6.

        Returns:
            RPCCPResult with all passes, collision, and synthesis.
        """
        start_time = time.time()
        history: List[PassResult] = []
        pass_types = ["naive", "critique", "unconstrained", "expand", "paradigm", "push"]

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"RPCCP — Recursive Precision Cognitive Calibration Protocol")
            print(f"{'='*70}")
            print(f"Query: {query}")
            print(f"Passes: {self.max_passes} + collision")
            print(f"Isolation: {self.isolation}")
            print(f"{'='*70}\n")

        # === PASSES 1 through max_passes ===
        for pass_num in range(1, self.max_passes + 1):
            model = self._get_model(pass_num)
            system = PASS_SYSTEMS.get(pass_num, PASS_SYSTEMS[6])
            pass_type = pass_types[pass_num - 1] if pass_num <= 6 else "extended"

            prompt = self._build_pass_prompt(pass_num, query, history)

            if self.verbose:
                print(f"--- Pass {pass_num}/{self.max_passes}: {pass_type.upper()} [{model.name}] ---")

            t0 = time.time()
            raw = model.generate(prompt, system=system)
            duration = time.time() - t0

            objective, dimensions, analysis = self._parse_response(raw)

            result = PassResult(
                pass_number=pass_num,
                pass_type=pass_type,
                prompt_sent=prompt,
                response=raw,
                model_used=model.name,
                objective_function=objective or f"Pass {pass_num} objective (unparsed)",
                dimensions_considered=dimensions or [f"Pass {pass_num} dimensions (unparsed)"],
                duration_seconds=round(duration, 1),
            )
            history.append(result)

            if self.verbose:
                print(f"  Objective: {result.objective_function}")
                print(f"  Dimensions: {', '.join(result.dimensions_considered[:5])}")
                print(f"  Duration: {duration:.1f}s")
                # Print first 200 chars of analysis
                preview = analysis[:200].replace("\n", " ")
                print(f"  Preview: {preview}...")
                print()

        # === COLLISION (Pass 7) ===
        collision_model = self._get_model(7)
        collision_system = PASS_SYSTEMS[7]
        collision_prompt = self._build_collision_prompt(query, history)

        if self.verbose:
            print(f"--- COLLISION [{collision_model.name}] ---")

        t0 = time.time()
        collision_raw = collision_model.generate(collision_prompt, system=collision_system)
        collision_duration = time.time() - t0

        collision = self._parse_collision(collision_raw, history)

        if self.verbose:
            print(f"  Type-2 Recursion Detected: {collision.type2_detected}")
            print(f"  Objective Evolution: {' → '.join(collision.objective_evolution[:5])}")
            print(f"  Real Question: {collision.real_question[:200]}")
            print(f"  Duration: {collision_duration:.1f}s")
            print()

        # === FINAL SYNTHESIS ===
        total_duration = time.time() - start_time
        models_used = list(set(p.model_used for p in history))

        # Build synthesis from collision
        synthesis = (
            f"RPCCP SYNTHESIS\n"
            f"{'='*50}\n"
            f"Original Query: {query}\n\n"
            f"Real Question Found: {collision.real_question}\n\n"
            f"Type-2 Recursion: {'YES' if collision.type2_detected else 'NO'}\n\n"
            f"Objective Evolution:\n"
        )
        for i, obj in enumerate(collision.objective_evolution):
            synthesis += f"  Pass {i+1}: {obj}\n"
        synthesis += f"\nCollision Output:\n{collision_raw}\n"

        result = RPCCPResult(
            query=query,
            passes=history,
            collision=collision,
            final_synthesis=synthesis,
            total_duration=round(total_duration, 1),
            models_used=models_used,
        )

        # Save to disk
        self._save_run(result)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"RPCCP COMPLETE — {total_duration:.1f}s total")
            print(f"Real Question: {collision.real_question[:200]}")
            print(f"Type-2 Recursion: {'DETECTED' if collision.type2_detected else 'not detected'}")
            print(f"Saved to: {self.output_dir}")
            print(f"{'='*70}\n")

        return result

    def _save_run(self, result: RPCCPResult):
        """Save complete run to disk as JSON."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = result.query[:40].replace(" ", "_").replace("?", "").replace("/", "_")
        filename = f"rpccp_{ts}_{slug}.json"

        data = {
            "query": result.query,
            "timestamp": result.timestamp,
            "total_duration": result.total_duration,
            "models_used": result.models_used,
            "type2_detected": result.collision.type2_detected if result.collision else False,
            "real_question": result.collision.real_question if result.collision else "",
            "objective_evolution": result.collision.objective_evolution if result.collision else [],
            "final_synthesis": result.final_synthesis,
            "passes": [
                {
                    "pass_number": p.pass_number,
                    "pass_type": p.pass_type,
                    "model_used": p.model_used,
                    "objective_function": p.objective_function,
                    "dimensions_considered": p.dimensions_considered,
                    "response": p.response,
                    "duration_seconds": p.duration_seconds,
                }
                for p in result.passes
            ],
        }

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[RPCCP] Saved run to {filepath}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Run RPCCP from command line."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="RPCCP — Recursive Precision Cognitive Calibration Protocol")
    parser.add_argument("query", nargs="?", help="The question to investigate")
    parser.add_argument("--model", default="qwen2.5:32b-instruct-q4_K_M", help="Default Ollama model")
    parser.add_argument("--passes", type=int, default=6, help="Number of passes (1-6)")
    parser.add_argument("--output", default="D:/DandyDon/rpccp_runs", help="Output directory")
    parser.add_argument("--diverse", action="store_true", help="Use different LOCAL models per pass")
    parser.add_argument("--cloud", action="store_true", help="Use Ollama Pro cloud models (fast, large)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if not args.query:
        # Interactive mode
        print("RPCCP — Enter your query (or 'quit' to exit):")
        args.query = input("> ").strip()
        if not args.query or args.query.lower() in ("quit", "exit", "q"):
            return

    models = {}
    if args.cloud:
        # Ollama Pro cloud — fast inference, massive models, no local RAM
        cloud_pool = [
            ("nemotron-3-super", "Naive solver — 120B MoE, 1M context"),
            ("kimi-k2-thinking", "Self-critic — deep reasoning"),
            ("qwen3.5:397b", "Unconstrained explorer — 397B parameters"),
            ("cogito-2.1:671b", "Dimension expander — 671B, broad knowledge"),
            ("deepseek-v3.2", "Paradigm breaker — strong reasoning"),
            ("nemotron-3-super", "Don's push — 120B MoE"),
            ("kimi-k2-thinking", "Collision engine — deep reasoning"),
        ]
        for i, (model_name, role) in enumerate(cloud_pool[:args.passes + 1], 1):
            models[i] = OllamaProModel(model_name)
        if not args.quiet:
            print("MODE: Ollama Pro Cloud — fast inference, massive models")
            print(f"Models: {', '.join(m for m, _ in cloud_pool[:args.passes + 1])}\n")
    elif args.diverse:
        # Assign different LOCAL models to different passes
        model_pool = [
            ("qwen2.5:32b-instruct-q4_K_M", "Naive solver"),
            ("hercules", "Self-critic"),
            ("phoenix", "Unconstrained explorer"),
            ("aaron", "Dimension expander"),
            ("davinci", "Paradigm breaker"),
            ("qwen2.5:32b-instruct-q4_K_M", "Don's push"),
            ("qwen2.5:32b-instruct-q4_K_M", "Collision engine"),
        ]
        for i, (model_name, role) in enumerate(model_pool[:args.passes + 1], 1):
            models[i] = OllamaModel(model_name)

    engine = RPCCP(
        models=models,
        default_model=OllamaModel(args.model),
        max_passes=args.passes,
        output_dir=args.output,
        verbose=not args.quiet,
    )

    result = engine.run(args.query)

    if args.quiet:
        print(f"\nReal Question: {result.collision.real_question}")
        print(f"Type-2 Recursion: {'YES' if result.collision.type2_detected else 'NO'}")


if __name__ == "__main__":
    main()
