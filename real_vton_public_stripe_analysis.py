#!/usr/bin/env python3
"""
公开真实条纹 VTON 输出的批量裁剪与 FDI 分析。

数据来源全部来自公开项目页、论文补充材料或公开项目视频帧，当前固定使用：
1. GP-VTON 官方 README 示例图
2. DCI-VTON 论文补充材料第 11 页
3. DCI-VTON 论文补充材料第 12 页
4. IDM-VTON 官方定性对比图 qualcmp_2
5. StableVITON、M&M VTO、GP-VTON、LaDI-VTON、IDM-VTON、GarDiff、VTON360 等公开材料的人工复核补充样本

脚本流程：
1. 从已固化到仓库的公开源图中裁剪 30 个真实条纹样本
2. 调用 FDI 分析器逐张计算 FDI / C-GOV / CURV / FFT 周期性
3. 输出 CSV、JSON 和总览拼图
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from PIL import Image, ImageDraw

from stripe_distortion_analyzer import analyze_stripe_distortion


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "test_images" / "real_vton_public_sources"
SAMPLE_DIR = ROOT / "test_images" / "real_vton_public_stripes"
OUTDIR = ROOT / "output"
ANALYSIS_DIR = OUTDIR / "real_vton_public_stripes"

SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DCI_METHODS = [
    "CP-VTON",
    "PF-AFN",
    "VITON-HD",
    "HR-VITON",
    "PaintByExample",
    "DCI-VTON",
]

IDM_METHODS = [
    "GP-VTON",
    "LaDI-VTON",
    "DCI-VTON",
    "StableVITON",
    "OOTDiffusion",
    "IDM-VTON",
]


@dataclass(frozen=True)
class CropGroup:
    """一组来自同一公开图源的裁剪定义。"""

    group_id: str
    source_file: str
    source_url: str
    source_note: str
    garment_desc: str
    stripe_type: str
    sample_prefix: str
    methods: list[str]
    boxes: list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class ExtraSample:
    """用于替换边界样本的单张补充图定义。"""

    sample_id: str
    source_path: str
    source_url: str
    source_note: str
    method: str
    target_garment: str
    stripe_type: str
    crop_box: tuple[int, int, int, int] | None = None


GROUPS = [
    CropGroup(
        group_id="gp_worldcup",
        source_file="gp_worldcup_vton.png",
        source_url="https://raw.githubusercontent.com/xiezhy6/GP-VTON/main/figures/worldcup_vton.png",
        source_note="GP-VTON README official qualitative example",
        garment_desc="Argentina-style football jersey with broad vertical stripes",
        stripe_type="broad vertical sports stripe",
        sample_prefix="gp_worldcup",
        methods=["GP-VTON"] * 6,
        boxes=[
            (302, 500, 671, 960),
            (673, 500, 1042, 960),
            (1044, 500, 1413, 960),
            (1415, 500, 1784, 960),
            (1786, 500, 2155, 960),
            (2157, 500, 2526, 960),
        ],
    ),
    CropGroup(
        group_id="dci_page11_row3",
        source_file="dci_supp_page11.png",
        source_url="https://arxiv.org/pdf/2308.06101.pdf",
        source_note="DCI-VTON paper supplementary, Figure 1 row 3",
        garment_desc="Multicolour horizontal striped knit top",
        stripe_type="multicolour horizontal stripe",
        sample_prefix="dci_row3",
        methods=DCI_METHODS,
        boxes=[
            (261, 604, 409, 790),
            (412, 604, 560, 790),
            (563, 604, 711, 790),
            (714, 604, 862, 790),
            (865, 604, 1013, 790),
            (1016, 604, 1164, 790),
        ],
    ),
    CropGroup(
        group_id="dci_page12_row4",
        source_file="dci_supp_page12.png",
        source_url="https://arxiv.org/pdf/2308.06101.pdf",
        source_note="DCI-VTON paper supplementary, Figure 2 row 4",
        garment_desc="Multicolour vertical striped dress",
        stripe_type="multicolour vertical stripe",
        sample_prefix="dci_row4",
        methods=DCI_METHODS,
        boxes=[
            (261, 804, 409, 987),
            (412, 804, 560, 987),
            (563, 804, 711, 987),
            (714, 804, 862, 987),
            (865, 804, 1013, 987),
            (1016, 804, 1164, 987),
        ],
    ),
    CropGroup(
        group_id="idm_qualcmp_row1",
        source_file="idm_qualcmp_2.png",
        source_url="https://idm-vton.github.io/",
        source_note="IDM-VTON project-page qualitative comparison, row 1",
        garment_desc="Black-white horizontal striped tank top",
        stripe_type="fine horizontal stripe",
        sample_prefix="idm_row1",
        methods=IDM_METHODS,
        boxes=[
            (320, 235, 890, 901),
            (945, 235, 1514, 901),
            (1568, 235, 2138, 901),
            (2192, 235, 2762, 901),
            (2816, 235, 3386, 901),
            (3440, 235, 3986, 901),
        ],
    ),
    CropGroup(
        group_id="idm_qualcmp_row3",
        source_file="idm_qualcmp_2.png",
        source_url="https://idm-vton.github.io/",
        source_note="IDM-VTON project-page qualitative comparison, row 3",
        garment_desc="Navy jersey with red-white horizontal sleeve and hem bands",
        stripe_type="sports band stripe",
        sample_prefix="idm_row3",
        methods=IDM_METHODS,
        boxes=[
            (291, 1795, 914, 2488),
            (915, 1795, 1538, 2488),
            (1539, 1795, 2162, 2488),
            (2163, 1795, 2786, 2488),
            (2787, 1795, 3410, 2488),
            (3411, 1795, 3999, 2488),
        ],
    ),
]

HISTORICAL_EXCLUDED_SAMPLE_IDS = {
    "dci_row3_01",
    "idm_row3_02",
    "idm_row3_03",
    "idm_row3_04",
    "idm_row3_05",
    "idm_row3_06",
}

REPLACEMENT_SAMPLES = {
    "gp_worldcup_01": ExtraSample(
        sample_id="mmvto_posed9_vertical",
        source_path="test_images/real_vton_public_sources/mmvto_posed_garment_vto_9.jpg",
        source_url="https://mmvto.github.io/static/images/posed_garment_vto/9.jpg",
        source_note="M&M VTO official project-page posed-garment result, manually cropped clean output",
        method="M&M VTO",
        target_garment="Short-sleeve shirt with vertical woven-look stripes",
        stripe_type="multicolour vertical stripe",
        crop_box=(3540, 70, 4685, 2045),
    ),
    "gp_worldcup_06": ExtraSample(
        sample_id="mmvto_posed11_orange",
        source_path="test_images/real_vton_public_sources/mmvto_posed_garment_vto_11.jpg",
        source_url="https://mmvto.github.io/static/images/posed_garment_vto/11.jpg",
        source_note="M&M VTO official project-page posed-garment result, manually cropped clean output",
        method="M&M VTO",
        target_garment="Orange polo shirt with broad horizontal stripe repeats",
        stripe_type="broad horizontal stripe",
        crop_box=(3660, 85, 4515, 2070),
    ),
    "dci_row3_03": ExtraSample(
        sample_id="mmvto_layflat7_blue_diag",
        source_path="test_images/real_vton_public_sources/mmvto_layflat_garment_vto_7.jpg",
        source_url="https://mmvto.github.io/static/images/layflat_garment_vto/7.jpg",
        source_note="M&M VTO official project-page lay-flat garment result, manually cropped clean output",
        method="M&M VTO",
        target_garment="Blue-white asymmetric striped knit top",
        stripe_type="diagonal broad stripe",
        crop_box=(7240, 25, 8205, 2115),
    ),
    "dci_row3_04": ExtraSample(
        sample_id="gp_vton_orange_horizontal",
        source_path="test_images/real_vton_public_sources/gp_vton_page7_300dpi.png",
        source_url="https://arxiv.org/pdf/2303.13756",
        source_note="GP-VTON public paper, qualitative comparison on page 7, manually cropped clean output",
        method="GP-VTON",
        target_garment="Orange-white horizontal striped knit top",
        stripe_type="medium horizontal stripe",
        crop_box=(1090, 1190, 1215, 1395),
    ),
    "dci_row3_06": ExtraSample(
        sample_id="mmvto_women1_graystripe",
        source_path="test_images/real_vton_public_sources/mmvto_person_identity_women1_4.jpg",
        source_url="https://mmvto.github.io/static/images/person_identity/women1_4.jpg",
        source_note="M&M VTO official project-page person-identity result, manually cropped clean output",
        method="M&M VTO",
        target_garment="Grey sweater with dark horizontal stripe repeats",
        stripe_type="medium horizontal sweater stripe",
        crop_box=(6170, 25, 7040, 2105),
    ),
    "dci_row4_01": ExtraSample(
        sample_id="mmvto_women2_navstripe",
        source_path="test_images/real_vton_public_sources/mmvto_person_identity_women2_4.jpg",
        source_url="https://mmvto.github.io/static/images/person_identity/women2_4.jpg",
        source_note="M&M VTO official project-page person-identity result, manually cropped clean output",
        method="M&M VTO",
        target_garment="Cream sweater with navy horizontal stripe repeats",
        stripe_type="bold horizontal sweater stripe",
        crop_box=(6170, 25, 7040, 2105),
    ),
    "dci_row4_02": ExtraSample(
        sample_id="gp_worldcup_06_replacement",
        source_path="test_images/real_vton_public_sources/gp_worldcup_vton.png",
        source_url="https://raw.githubusercontent.com/xiezhy6/GP-VTON/main/figures/worldcup_vton.png",
        source_note="GP-VTON README official qualitative example, manually cropped clean output",
        method="GP-VTON",
        target_garment="Argentina-style football jersey with broad vertical stripes",
        stripe_type="broad vertical sports stripe",
        crop_box=(2157, 500, 2526, 960),
    ),
    "dci_row4_03": ExtraSample(
        sample_id="ladi_vton_fine_horizontal",
        source_path="test_images/real_vton_public_sources/ladi_vton_page15_300dpi.png",
        source_url="https://arxiv.org/pdf/2305.13501",
        source_note="LaDI-VTON public paper, qualitative comparison on page 15, manually cropped clean output",
        method="LaDI-VTON",
        target_garment="White-black fine horizontal striped top",
        stripe_type="fine horizontal stripe",
        crop_box=(1865, 1950, 2250, 2475),
    ),
    "idm_row1_02": ExtraSample(
        sample_id="gp_vton_fine_stripe_tank",
        source_path="test_images/real_vton_public_sources/gp_vton_page7_300dpi.png",
        source_url="https://arxiv.org/pdf/2303.13756",
        source_note="GP-VTON public paper, qualitative comparison on page 7, manually cropped clean output",
        method="GP-VTON",
        target_garment="White-black fine horizontal striped tank top",
        stripe_type="fine horizontal stripe",
        crop_box=(2150, 1190, 2285, 1395),
    ),
    "idm_row1_05": ExtraSample(
        sample_id="idm_vton_crossing_stripe",
        source_path="test_images/real_vton_public_sources/idm_qualcmp_3.png",
        source_url="https://idm-vton.github.io/",
        source_note="IDM-VTON official project-page qualitative comparison, manually cropped clean output",
        method="IDM-VTON",
        target_garment="Red-blue crossing-stripe top",
        stripe_type="crossing stripe",
        crop_box=(3410, 1040, 3990, 1845),
    ),
    "idm_row1_06": ExtraSample(
        sample_id="gardiff_horizontal_top",
        source_path="test_images/real_vton_public_sources/gardiff_page11_300dpi.png",
        source_url="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06545.pdf",
        source_note="GarDiff ECCV 2024 public paper, qualitative comparison on page 11, manually cropped clean output",
        method="GarDiff",
        target_garment="White-black horizontal striped top",
        stripe_type="medium horizontal stripe",
        crop_box=(1805, 1480, 1980, 1705),
    ),
    "idm_row3_01": ExtraSample(
        sample_id="vton360_rugby_stripe",
        source_path="test_images/real_vton_public_sources/vton360_qualitative_comparison_1_frame277.jpg",
        source_url="https://scnuhealthy.github.io/VTON360/videos/Qualitative_comparison_1.mp4",
        source_note="VTON360 official project-page qualitative-comparison video frame, Ours column, manually cropped clean output",
        method="VTON360",
        target_garment="Rugby-style shirt with broad horizontal bands",
        stripe_type="broad horizontal sports stripe",
        crop_box=(675, 515, 925, 940),
    ),
}

EXTRA_SAMPLES = [
    ExtraSample(
        sample_id="stableviton_public_01",
        source_path="test_images/real_vton_stableviton/subject1_runway_female.png",
        source_url="https://rlawjdghek.github.io/StableVITON/",
        source_note="StableVITON official project-page striped-garment example",
        method="StableVITON",
        target_garment="Black-white horizontal stripe top",
        stripe_type="bold horizontal stripe",
    ),
    ExtraSample(
        sample_id="stableviton_public_02",
        source_path="test_images/real_vton_stableviton/subject2_stage_male.png",
        source_url="https://rlawjdghek.github.io/StableVITON/",
        source_note="StableVITON official project-page striped-garment example",
        method="StableVITON",
        target_garment="Black-white horizontal stripe top",
        stripe_type="bold horizontal stripe",
    ),
    ExtraSample(
        sample_id="idm_row1_06_replacement",
        source_path="test_images/real_vton_public_sources/idm_qualcmp_2.png",
        source_url="https://idm-vton.github.io/",
        source_note="IDM-VTON project-page qualitative comparison, row 1, IDM-VTON column, manually cropped clean output",
        method="IDM-VTON",
        target_garment="Black-white horizontal striped tank top",
        stripe_type="fine horizontal stripe",
        crop_box=(3440, 235, 3986, 901),
    ),
    ExtraSample(
        sample_id="stableviton_public_04",
        source_path="test_images/real_vton_stableviton/subject4_event_female.png",
        source_url="https://rlawjdghek.github.io/StableVITON/",
        source_note="StableVITON official project-page striped-garment example",
        method="StableVITON",
        target_garment="Black-white horizontal stripe top",
        stripe_type="bold horizontal stripe",
        crop_box=(0, 55, 620, 660),
    ),
    ExtraSample(
        sample_id="stableviton_public_05",
        source_path="test_images/real_vton_stableviton/subject5_redcarpet_male.png",
        source_url="https://rlawjdghek.github.io/StableVITON/",
        source_note="StableVITON official project-page striped-garment example",
        method="StableVITON",
        target_garment="Black-white horizontal stripe top",
        stripe_type="bold horizontal stripe",
        crop_box=(0, 85, 620, 640),
    ),
    ExtraSample(
        sample_id="dci_teaser_ours",
        source_path="test_images/real_vton_public_sources/dci_teaser.jpg",
        source_url="https://arxiv.org/pdf/2308.06101.pdf",
        source_note="DCI-VTON official teaser panel (c) Ours",
        method="DCI-VTON",
        target_garment="Multicolour horizontal striped top",
        stripe_type="fine horizontal stripe",
        crop_box=(2940, 120, 3913, 1280),
    ),
]


def categorize_fdi(fdi: float) -> str:
    """沿用论文中的临时筛查分段标签。"""
    if fdi < 15:
        return "LR"
    if fdi < 30:
        return "IS"
    if fdi < 45:
        return "VI"
    return "HR"


def format_one_decimal(value: float) -> str:
    """按论文习惯用 half-up 保留一位小数。"""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def crop_public_samples() -> list[dict]:
    """从公开源图裁出 30 个真实条纹样本，并返回元数据。"""
    records: list[dict] = []

    for group in GROUPS:
        source_path = SOURCE_DIR / group.source_file
        if not source_path.exists():
            raise FileNotFoundError(f"缺少公开源图：{source_path}")

        image = Image.open(source_path).convert("RGB")
        for index, (method_name, box) in enumerate(zip(group.methods, group.boxes), start=1):
            sample_id = f"{group.sample_prefix}_{index:02d}"
            sample_path = SAMPLE_DIR / f"{sample_id}.png"
            image.crop(box).save(sample_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(sample_path),
                    "source_group": group.group_id,
                    "source_file": group.source_file,
                    "source_url": group.source_url,
                    "source_note": group.source_note,
                    "method": method_name,
                    "target_garment": group.garment_desc,
                    "stripe_type": group.stripe_type,
                    "crop_box": list(box),
                }
            )

    return records


def append_extra_samples(records: list[dict]) -> list[dict]:
    """补入人工复核后更干净的替换样本。"""
    filtered: list[dict] = []
    for item in records:
        sample_id = item["sample_id"]
        if sample_id in HISTORICAL_EXCLUDED_SAMPLE_IDS:
            continue
        if sample_id in REPLACEMENT_SAMPLES:
            filtered.append(materialize_extra_sample(REPLACEMENT_SAMPLES[sample_id], "manual_diversity_replacement"))
            continue
        filtered.append(item)

    for extra in EXTRA_SAMPLES:
        filtered.append(materialize_extra_sample(extra, "manual_replacement"))

    return filtered


def materialize_extra_sample(extra: ExtraSample, source_group: str) -> dict:
    """读取单张补充源图，按裁剪框输出为正式样本记录。"""
    source_path = ROOT / extra.source_path
    if not source_path.exists():
        raise FileNotFoundError(f"缺少替换样本源图：{source_path}")

    image = Image.open(source_path).convert("RGB")
    if extra.crop_box is not None:
        image = image.crop(extra.crop_box)

    sample_path = SAMPLE_DIR / f"{extra.sample_id}.png"
    image.save(sample_path)
    return {
        "sample_id": extra.sample_id,
        "image_path": str(sample_path),
        "source_group": source_group,
        "source_file": source_path.name,
        "source_url": extra.source_url,
        "source_note": extra.source_note,
        "method": extra.method,
        "target_garment": extra.target_garment,
        "stripe_type": extra.stripe_type,
        "crop_box": list(extra.crop_box) if extra.crop_box is not None else None,
    }


def ensure_no_duplicate_samples(records: list[dict]) -> None:
    """检查是否误把同一张输出图重复纳入样本集。"""
    seen: dict[str, str] = {}
    for item in records:
        image_bytes = Path(item["image_path"]).read_bytes()
        digest = hashlib.sha1(image_bytes).hexdigest()
        if digest in seen:
            raise ValueError(
                f'检测到重复样本：{item["sample_id"]} 与 {seen[digest]} 的图像内容完全一致'
            )
        seen[digest] = item["sample_id"]


def build_contact_sheet(results: list[dict]) -> Path:
    """生成 30 个真实样本的总览拼图，便于人工复核。"""
    thumb_w, thumb_h = 220, 280
    caption_h = 48
    cols = 5
    rows = (len(results) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption_h)), "#f4f4f4")

    for idx, item in enumerate(results):
        image = Image.open(item["image_path"]).convert("RGB")
        image.thumbnail((thumb_w - 16, thumb_h - 16))

        cell = Image.new("RGB", (thumb_w, thumb_h + caption_h), "white")
        x = (thumb_w - image.width) // 2
        y = 8
        cell.paste(image, (x, y))

        draw = ImageDraw.Draw(cell)
        caption = f'{item["sample_id"]} | {item["method"]}\nFDI={format_one_decimal(item["fdi"])}'
        draw.text((8, thumb_h + 4), caption, fill="black")

        row = idx // cols
        col = idx % cols
        sheet.paste(cell, (col * thumb_w, row * (thumb_h + caption_h)))

    out_path = OUTDIR / "public_real_vton_stripe_panel.png"
    sheet.save(out_path)
    return out_path


def write_csv(results: list[dict], csv_path: Path) -> None:
    """保存 CSV，方便后续直接导入表格。"""
    fieldnames = [
        "sample_id",
        "method",
        "target_garment",
        "stripe_type",
        "fdi",
        "c_gov",
        "curv",
        "periodicity",
        "category",
        "source_group",
        "source_file",
        "source_note",
        "source_url",
        "image_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fieldnames})


def summarize_results(results: list[dict]) -> dict:
    """输出整体统计和按方法统计。"""
    fdi_values = [item["fdi"] for item in results]
    summary = {
        "sample_count": len(results),
        "mean_fdi": round(statistics.mean(fdi_values), 2),
        "std_fdi": round(statistics.pstdev(fdi_values), 2),
        "min_fdi": round(min(fdi_values), 2),
        "max_fdi": round(max(fdi_values), 2),
        "by_method": {},
    }

    methods = sorted({item["method"] for item in results})
    for method in methods:
        subset = [item["fdi"] for item in results if item["method"] == method]
        summary["by_method"][method] = {
            "count": len(subset),
            "mean_fdi": round(statistics.mean(subset), 2),
            "min_fdi": round(min(subset), 2),
            "max_fdi": round(max(subset), 2),
        }

    return summary


def main() -> None:
    records = crop_public_samples()
    records = append_extra_samples(records)
    ensure_no_duplicate_samples(records)
    results: list[dict] = []

    for item in records:
        fdi, roi_gov, curv, periodicity = analyze_stripe_distortion(
            image_path=item["image_path"],
            output_dir=str(ANALYSIS_DIR),
            blur_ksize=3,
            gov_window_size=7,
            mag_threshold=15.0,
            canny_low=15.0,
            canny_high=50.0,
            min_contour_len=20,
            w_gov=0.65,
            w_curv=0.35,
            use_skin_suppression=True,
            use_texture_filter=True,
        )
        enriched = dict(item)
        enriched.update(
            {
                "fdi": round(fdi, 2),
                "c_gov": round(roi_gov, 4),
                "curv": round(curv, 4),
                "periodicity": round(periodicity, 3),
                "category": categorize_fdi(fdi),
            }
        )
        results.append(enriched)

    summary = summarize_results(results)

    json_path = OUTDIR / "public_real_vton_stripe_results.json"
    csv_path = OUTDIR / "public_real_vton_stripe_results.csv"
    panel_path = build_contact_sheet(results)

    payload = {"summary": summary, "samples": results}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(results, csv_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {panel_path}")


if __name__ == "__main__":
    main()
