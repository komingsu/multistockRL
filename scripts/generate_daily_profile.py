from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from textwrap import dedent

DATA_PATH = Path("data/proc/daily_with_indicators.csv")
OUTPUT_PATH = Path("docs/daily_with_indicators_profile.md")

COLUMN_DESCRIPTIONS = {
    "symbol": "숫자로 표현된 한국거래소 종목 코드.",
    "date": "거래일 (YYYY-MM-DD) 문자열.",
    "open": "시가 (KRW).",
    "high": "고가 (KRW).",
    "low": "저가 (KRW).",
    "close": "종가 (KRW).",
    "volume": "거래량 (주식 수).",
    "value": "거래대금 (KRW).",
    "adj_flag": "수정주가 적용 여부 (본 데이터는 모두 True).",
    "name": "종목명 (한글).",
    "market": "시장 구분 (KOSPI, KOSPI200, KOSDAQ, KONEX, KSQ150).",
    "industry": "KRX 산업 분류명.",
    "market_cap": "시가총액 (단위: 억 KRW 추정).",
    "prdy_close": "전일 종가 (해당 종목 첫 거래일에는 결측).",
    "change": "전일 대비 종가 절대 변동폭 (KRW).",
    "change_rate": "전일 대비 종가 변동률 (퍼센트 포인트).",
    "prsn_buy_val_ratio": "개인 투자자 매수 대금 비중 (0~1).",
    "prsn_sell_val_ratio": "개인 투자자 매도 대금 비중 (0~1).",
    "prsn_net_val_ratio": "개인 투자자 순매수 대금 비중 (0~1).",
    "prsn_buy_vol_ratio": "개인 투자자 매수 물량 비중 (0~1).",
    "prsn_sell_vol_ratio": "개인 투자자 매도 물량 비중 (0~1).",
    "prsn_net_vol_ratio": "개인 투자자 순매수 물량 비중 (0~1).",
    "frgn_buy_val_ratio": "외국인 매수 대금 비중 (0~1).",
    "frgn_sell_val_ratio": "외국인 매도 대금 비중 (0~1).",
    "frgn_net_val_ratio": "외국인 순매수 대금 비중 (0~1).",
    "frgn_buy_vol_ratio": "외국인 매수 물량 비중 (0~1).",
    "frgn_sell_vol_ratio": "외국인 매도 물량 비중 (0~1).",
    "frgn_net_vol_ratio": "외국인 순매수 물량 비중 (0~1).",
    "orgn_buy_val_ratio": "기관 매수 대금 비중 (0~1).",
    "orgn_sell_val_ratio": "기관 매도 대금 비중 (0~1).",
    "orgn_net_val_ratio": "기관 순매수 대금 비중 (0~1).",
    "orgn_buy_vol_ratio": "기관 매수 물량 비중 (0~1).",
    "orgn_sell_vol_ratio": "기관 매도 물량 비중 (0~1).",
    "orgn_net_vol_ratio": "기관 순매수 물량 비중 (0~1).",
    "ma_5": "5일 이동평균 (종가 기준).",
    "ma_10": "10일 이동평균.",
    "ma_20": "20일 이동평균.",
    "dist_ma5": "종가 대비 5일 이동평균 상대 거리.",
    "dist_ma10": "종가 대비 10일 이동평균 상대 거리.",
    "dist_ma20": "종가 대비 20일 이동평균 상대 거리.",
    "ma10_slope": "10일 이동평균 기울기 (추세 지표).",
    "ma20_slope": "20일 이동평균 기울기.",
    "atr_5": "5일 Average True Range.",
    "atr_14": "14일 Average True Range.",
    "rv_10": "10일 실현 변동성.",
    "rv_20": "20일 실현 변동성.",
    "range_pct": "일중 고저 변동폭 비율.",
    "gap_pct": "시가와 전일 종가 사이 갭 비율.",
}


def fmt_count(value: int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_float(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    abs_val = abs(float(value))
    if abs_val >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{value:,.0f}"
    if abs_val >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report() -> str:
    df = pd.read_csv(DATA_PATH)

    rows = len(df)
    symbol_count = df["symbol"].nunique()
    date_count = df["date"].nunique()
    start_date = df["date"].min()
    end_date = df["date"].max()

    per_market = (
        df.groupby("market")["symbol"].nunique().sort_values(ascending=False)
    )
    market_lines = [f"- {market}: {count}개 종목" for market, count in per_market.items()]

    symbol_ranges = df.groupby("symbol").agg(
        name=("name", "first"),
        market=("market", "first"),
        industry=("industry", "first"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        rows=("date", "count"),
    ).reset_index()
    symbol_ranges["industry"] = symbol_ranges["industry"].fillna("(미기재)")

    symbol_rows = []
    for _, record in symbol_ranges.sort_values("symbol").iterrows():
        symbol_rows.append(
            [
                str(int(record["symbol"])),
                record["name"],
                record["market"],
                record["industry"],
                fmt_count(record["rows"]),
                record["start_date"],
                record["end_date"],
            ]
        )

    identifier_cols = ["symbol", "date", "name", "market", "industry", "adj_flag"]
    numeric_cols = [col for col in df.columns if col not in identifier_cols]

    cat_rows: list[list[str]] = []
    for col in identifier_cols:
        series = df[col]
        distinct = series.nunique(dropna=True)
        missing = series.isna().sum()
        examples = pd.Series(series.dropna().astype(str).unique()[:3]).tolist()
        example_str = ", ".join(examples)
        desc = COLUMN_DESCRIPTIONS.get(col, "")
        cat_rows.append(
            [
                f"`{col}`",
                str(series.dtype),
                fmt_count(distinct),
                fmt_count(missing),
                fmt_pct(missing / len(series) * 100),
                example_str,
                desc,
            ]
        )

    numeric_rows: list[list[str]] = []
    for col in numeric_cols:
        series = df[col]
        missing = series.isna().sum()
        info = {
            "non_null": len(series) - missing,
            "missing": missing,
            "missing_pct": missing / len(series) * 100,
            "min": series.min(skipna=True),
            "max": series.max(skipna=True),
            "mean": series.mean(skipna=True),
            "std": series.std(skipna=True),
            "p05": series.quantile(0.05),
            "p95": series.quantile(0.95),
        }
        desc = COLUMN_DESCRIPTIONS.get(col, "")
        numeric_rows.append(
            [
                f"`{col}`",
                str(series.dtype),
                fmt_count(info["non_null"]),
                fmt_count(info["missing"]),
                fmt_pct(info["missing_pct"]),
                fmt_float(info["min"]),
                fmt_float(info["p05"]),
                fmt_float(info["mean"]),
                fmt_float(info["p95"]),
                fmt_float(info["max"]),
                fmt_float(info["std"]),
                desc,
            ]
        )

    missing_counts = df.isna().sum()
    lookback_features = [
        "prdy_close",
        "gap_pct",
        "ma_5",
        "ma_10",
        "ma_20",
        "dist_ma5",
        "dist_ma10",
        "dist_ma20",
        "ma10_slope",
        "ma20_slope",
        "atr_5",
        "atr_14",
        "rv_10",
        "rv_20",
    ]
    lookback_lines = []
    for col in lookback_features:
        missing = int(missing_counts.get(col, 0))
        if missing:
            per_symbol = missing / symbol_count
            reason = "이동평균/변동성 계산을 위한 워밍업 구간"
            if col in {"prdy_close", "gap_pct"}:
                reason = "종목 첫 거래일에는 전일 데이터 부재"
            lookback_lines.append(
                f"- `{col}`: {fmt_count(missing)}건 결측 (종목당 {per_symbol:.0f}일) - {reason}."
            )

    industry_missing_symbols = (
        df.loc[df["industry"].isna(), "symbol"].unique().tolist()
    )
    industry_line = "- `industry`: 1,141건 결측 - 종목 코드 " + ", ".join(
        str(int(code)) for code in industry_missing_symbols
    ) + " 전 구간에서 산업 분류 부재."

    duplicate_rows = int(df.duplicated(subset=["symbol", "date"]).sum())

    overview_block = dedent(
        f"""
        # `daily_with_indicators.csv` 데이터 프로파일

        ## 스냅샷
        - 총 {fmt_count(rows)}개 row, {symbol_count}개 종목, {date_count}개 거래일.
        - 데이터 기간: {start_date} ~ {end_date} ({date_count} 거래일).
        - 중복 `(symbol, date)` 조합: {duplicate_rows}건.

        ## 시장 구성
        """
    ).strip()

    cat_table = to_markdown(
        ["Column", "Type", "Distinct", "Missing", "Missing %", "Examples", "설명"],
        cat_rows,
    )
    num_table = to_markdown(
        [
            "Column",
            "Type",
            "Non-null",
            "Missing",
            "Missing %",
            "Min",
            "P05",
            "Mean",
            "P95",
            "Max",
            "Std",
            "설명",
        ],
        numeric_rows,
    )
    symbol_table = to_markdown(
        ["Symbol", "Name", "Market", "Industry", "Rows", "Start", "End"],
        symbol_rows,
    )

    report_parts = [
        overview_block,
        "\n".join(market_lines),
        "\n\n## 종목 메타데이터",
        symbol_table,
        "\n\n## 식별자 및 범주형 컬럼",
        cat_table,
        "\n\n## 수치형 피처 요약",
        num_table,
        "\n\n## 결측치 메모",
        "\n".join(lookback_lines + [industry_line]),
    ]

    report = "\n\n".join(part for part in report_parts if part)
    return report + "\n"


def main() -> None:
    report = build_report()
    OUTPUT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
