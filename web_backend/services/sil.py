"""Task-friendly adapter around the existing GSPN-MC SIL engine."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def validate(config: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    from sil_validation import GSPN_MooN_MGL, SimParams, sil_from_pfd

    m = int(config.get("m", 2))
    n = int(config.get("n", 4))
    lam_fit = float(config.get("lambda_fit", 111.11))
    ti = float(config.get("ti", 8760))
    mrt = float(config.get("mrt", 8))
    nsim = int(config.get("nsim", 500))
    years = int(config.get("years", 10000))
    mode = str(config.get("ccf_mode", "total"))
    if not (1 <= m <= n <= 10):
        raise ValueError("表决架构必须满足 1 ≤ M ≤ N ≤ 10")
    if not all(np.isfinite(value) for value in (lam_fit, ti, mrt)) or lam_fit < 0.01 or lam_fit > 100000 or ti < 1 or ti > 100000 or mrt < 0 or mrt > 10000 or not (1 <= nsim <= 2000) or years < 1001 or years > 100000:
        raise ValueError("请检查失效率、测试间隔、修复时间、仿真次数和仿真年数")

    lam = lam_fit * 1e-9
    if mode == "total":
        beta = float(config.get("total_beta", 0.1))
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("全局β必须在 0 到 1 之间（含边界）")
        # CS 端的“全局共因”入口会把 β 均分到 β2...βN，然后统一
        # 通过 MGL 仿真器执行；BS 保持相同的模型和分布口径。
        beta_list = [beta / (n - 1)] * (n - 1) if n > 1 else []
        params = SimParams(N=n, M=m, TI=ti, MRT=mrt, LAMBDA_DU=lam, BETA1=1.0 - beta, beta_list=beta_list, SIM_YEARS=years, WARMUP_YEARS=max(1000, years // 10), NUM_SIM=nsim)
        simulator_type = GSPN_MooN_MGL
        ccf_details = {"mode": "全局共因 (Total β)", "total_beta": beta}
    else:
        raw_beta = config.get("partial_betas", {})
        beta_list = [float(raw_beta.get(str(k), raw_beta.get(k, 0.0))) for k in range(2, n + 1)]
        if any(not np.isfinite(beta) or beta < 0 or beta > 1 for beta in beta_list) or sum(beta_list) >= 1:
            raise ValueError("部分β必须在 0 到 1 之间，且总和小于 1")
        params = SimParams(N=n, M=m, TI=ti, MRT=mrt, LAMBDA_DU=lam, BETA1=1.0 - sum(beta_list), beta_list=beta_list, SIM_YEARS=years, WARMUP_YEARS=max(1000, years // 10), NUM_SIM=nsim)
        simulator_type = GSPN_MooN_MGL
        ccf_details = {"mode": "部分共因 (Partial β)", "partial_betas": {str(index + 2): value for index, value in enumerate(beta_list)}}

    log(f"开始GSPN-MC仿真：{m}oo{n}，共 {nsim} 次。")
    results: list[float] = []
    report_interval = max(1, nsim // 20)
    for index in range(nsim):
        results.append(float(simulator_type(params).simulate()))
        if (index + 1) % report_interval == 0 or index + 1 == nsim:
            log(f"仿真进度：{index + 1}/{nsim}")
    samples = np.asarray(results, dtype=float)
    mean_pfd = float(np.mean(samples))
    std_pfd = float(np.std(samples, ddof=1)) if nsim > 1 else 0.0
    # CS 端使用 matplotlib 的 30 个等宽分箱和 density=True。返回密度
    # 与分箱中心，BS 端即可按同一统计口径绘制直方图；同时保留 counts
    # 兼容已有调用方。
    counts, edges = np.histogram(samples, bins=30)
    density, _ = np.histogram(samples, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "architecture": {"m": m, "n": n},
        "lambda_fit": lam_fit,
        "ti": ti,
        "mrt": mrt,
        "simulations": nsim,
        "years": years,
        "ccf": ccf_details,
        "pfdavg": mean_pfd,
        "std": std_pfd,
        "confidence_interval": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "sil": sil_from_pfd(mean_pfd),
        "histogram": {
            "edges": edges.tolist(),
            "centers": centers.tolist(),
            "counts": counts.tolist(),
            "density": density.tolist(),
        },
    }
