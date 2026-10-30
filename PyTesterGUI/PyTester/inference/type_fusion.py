"""
TypeFusion

Corrected to fuse:
- constructor args (from ASTInspector)
- method args (from ASTInspector)
- function args (from ASTInspector)
- semantic argument types
- dynamic argument types
"""

from __future__ import annotations

from typing import Dict, Any, Optional


class TypeFusion:
    """
    Fuse static, semantic, and dynamic inference results.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def merge(
        self,
        static_info: Dict[str, Any],
        semantic_info: Dict[str, Any],
        dynamic_info: Dict[str, Any],
    ) -> Dict[str, Any]:

        fused: Dict[str, Any] = {}

        all_names = (
            set(static_info.keys())
            | set(semantic_info.keys())
            | set(dynamic_info.keys())
        )

        for name in all_names:
            s = static_info.get(name, {})
            m = semantic_info.get(name, {})
            d = dynamic_info.get(name, {})

            fused_args = self._fuse_args(
                static_args=s.get("args", {}),
                semantic_args=m.get("arg_semantics", {}),
                dynamic_args=d.get("arg_types", {}),
            )

            fused_return = self._fuse_return(
                static_ret=s.get("return"),
                semantic_ret=m.get("return_semantics"),
                dynamic_ret=d.get("safe_return_type"),
            )

            fused_behavior = self._fuse_behavior(
                static_props=s.get("properties", {}),
                semantic_behavior=m.get("behavior", {}),
            )

            fused[name] = {
                "kind": s.get("kind"),
                "args": fused_args,
                "return": fused_return,
                "semantic_return": m.get("return_semantics"),
                "dynamic_return": d.get("safe_return_type"),
                "intent": m.get("intent"),
                "behavior": fused_behavior,
                "confidence": self._compute_confidence(
                    static_ret=s.get("return"),
                    semantic_ret=m.get("return_semantics"),
                    dynamic_ret=d.get("safe_return_type"),
                ),
            }

        return fused

    # ------------------------------------------------------------
    # Argument fusion
    # ------------------------------------------------------------
    def _fuse_args(
        self,
        static_args: Dict[str, Any],
        semantic_args: Dict[str, Any],
        dynamic_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge static, semantic, and dynamic argument types.

        Priority:
        1. static (most precise)
        2. semantic
        3. dynamic
        """

        fused = dict(static_args)

        # semantic types
        for arg, sem_type in semantic_args.items():
            if arg not in fused or fused[arg] is None:
                fused[arg] = sem_type

        # dynamic types
        for arg, dyn_type in dynamic_args.items():
            if arg not in fused or fused[arg] is None:
                fused[arg] = dyn_type

        return fused

    # ------------------------------------------------------------
    # Return fusion
    # ------------------------------------------------------------
    def _fuse_return(
        self,
        static_ret: Optional[str],
        semantic_ret: Optional[str],
        dynamic_ret: Optional[str],
    ) -> Optional[str]:

        if static_ret:
            return static_ret
        if semantic_ret and semantic_ret != "unknown":
            return semantic_ret
        if dynamic_ret and dynamic_ret != "unknown":
            return dynamic_ret
        return None

    # ------------------------------------------------------------
    # Behavior fusion
    # ------------------------------------------------------------
    def _fuse_behavior(
        self,
        static_props: Dict[str, Any],
        semantic_behavior: Dict[str, Any],
    ) -> Dict[str, Any]:

        fused = dict(static_props)
        for k, v in semantic_behavior.items():
            fused[k] = v
        return fused

    # ------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------
    def _compute_confidence(
        self,
        static_ret: Optional[str],
        semantic_ret: Optional[str],
        dynamic_ret: Optional[str],
    ) -> float:

        score = 0.0

        if static_ret:
            score += 0.5
        if semantic_ret and semantic_ret != "unknown":
            score += 0.3
        if dynamic_ret and dynamic_ret != "unknown":
            score += 0.2

        return min(score, 1.0)
