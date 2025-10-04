from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ScaleSentryConfig:
    threshold: float = 0.25                 # 相对偏差阈值（例如 0.25 表示 25%）
    window_steps: int = 50                  # 进入保护后持续的步数
    grad_clip_during_sentry: float = 0.5
    p_weight_drop_on_sentry: float = 0.2
    ema_momentum: float = 0.9               # EMA 平滑系数
    grace_steps: int = 500                  # 训练早期宽限期
    min_bad_streak: int = 5                 # 连续异常步数达到这个值才触发
    ignore_scale_when_near_one: bool = True # 当 s≈1 时不把它作为触发信号
    near_one_tol: float = 1e-3              # 判定“≈1”的容差

class ScaleSentry:
    """
    Monitors scale stability via alpha (D2 ~ alpha * d_T + beta) and an external scale_ref (distance-domain).
    Uses EMA references and relative deviation; triggers temporary tightening if sustained deviation occurs.
    """

    def __init__(self, cfg: Optional[ScaleSentryConfig] = None):
        self.cfg = cfg or ScaleSentryConfig()
        self.alpha_ref: Optional[float] = None    # EMA reference for alpha
        self.scale_ref: Optional[float] = None    # EMA reference for distance-domain scale (optional)
        self._bad_streak: int = 0
        self._active_until: int = -1
        self._now_step: int = 0
        self.hits: int = 0
        self._last_active_state: bool = False     # 仅用于“刚触发”判断

    def _ema_update(self, ref: Optional[float], x: float) -> float:
        m = float(self.cfg.ema_momentum)
        return x if ref is None else (m * float(ref) + (1.0 - m) * float(x))

    def _rel_dev(self, x: float, ref: Optional[float]) -> float:
        if ref is None:
            return 0.0
        refv = max(abs(float(ref)), 1e-12)
        return abs(float(x) / refv - 1.0)

    def update(self, step: int, alpha: float, scale_ref: Optional[float] = None) -> Dict[str, float]:
        self._now_step = int(step)

        # --- 先更新 EMA 参考 ---
        self.alpha_ref = self._ema_update(self.alpha_ref, float(alpha))
        if scale_ref is not None:
            self.scale_ref = self._ema_update(self.scale_ref, float(scale_ref))

        # --- 计算相对偏差 ---
        dev_alpha = self._rel_dev(float(alpha), self.alpha_ref)
        dev_scale = 0.0
        use_scale = (scale_ref is not None) and (self.scale_ref is not None)
        if use_scale:
            # 若 s 一直被 clamp 到 ~1，则忽略它作为触发信号，避免“永远冲突”
            if self.cfg.ignore_scale_when_near_one and abs(float(self.scale_ref) - 1.0) < self.cfg.near_one_tol:
                use_scale = False
            else:
                dev_scale = self._rel_dev(float(scale_ref), self.scale_ref)

        # --- 早期宽限期：直接不触发 ---
        if self._now_step < int(self.cfg.grace_steps):
            dev_flag = False
            self._bad_streak = 0
        else:
            dev_flag = (dev_alpha > self.cfg.threshold) or (use_scale and dev_scale > self.cfg.threshold)
            if dev_flag:
                self._bad_streak += 1
            else:
                self._bad_streak = max(0, self._bad_streak - 1)

        # --- 进入保护的判定（带最小连续步数）---
        just_triggered = False
        if (self._bad_streak >= int(self.cfg.min_bad_streak)) and (self._now_step >= self._active_until):
            self._active_until = self._now_step + int(self.cfg.window_steps)
            self.hits += 1
            just_triggered = True

        active_now = self._now_step < self._active_until

        adj = {
            "grad_clip": float(self.cfg.grad_clip_during_sentry) if active_now else 0.0,
            "p_weight_scale": float(1.0 - self.cfg.p_weight_drop_on_sentry) if active_now else 1.0,
            "active": 1.0 if active_now else 0.0,
            "just_triggered": 1.0 if just_triggered else 0.0,
            # 可选：把观测到的偏差返回，便于日志
            "dev_alpha": float(dev_alpha),
            "dev_scale": float(dev_scale) if use_scale else 0.0,
        }
        self._last_active_state = active_now
        return adj