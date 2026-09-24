import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import streamlit as st
import random

import sys
import toml


config_file_path = sys.argv[1] if len(sys.argv) >= 2 else "config/config.toml"
assert os.path.exists(config_file_path), f"Config file not found: {config_file_path}"

# Load configuration from TOML file
with open(config_file_path, "r") as f:
    config = toml.load(f)


# --------------------------
# Config (adjust as needed)
# --------------------------
DEFAULT_IMAGES_DIR = config['bbox_approver']['image_folder_path']
DEFAULT_COCO_PATH  = config['bbox_approver']['coco_annotations_path_input']
AUTOSAVE_PATH      = config['bbox_approver']['coco_annotations_path_autosave']
FILTERED_COCO_OUT  = config['bbox_approver']['coco_annotations_path_processed']


# --------------------------
# Helpers
# --------------------------
def load_coco(coco_path: Path) -> Dict:
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    # Basic checks
    for k in ["images", "annotations", "categories"]:
        if k not in coco:
            raise ValueError(f"COCO file missing key: {k}")
    return coco


def index_coco(coco: Dict) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], Dict[int, str]]:
    """Return:
      img_by_id: {image_id -> image_dict}
      anns_by_img: {image_id -> [ann_dict, ...]}
      cat_name: {category_id -> category_name}
    """
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    
    cat_name = {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}

    return img_by_id, anns_by_img, cat_name


def draw_boxes(
    image_path: Path,
    anns: List[Dict],
    decisions: Dict[int, str],  # {ann_id: "approved"/"rejected"/"undecided"}
    cat_name: Dict[int, str],
    line_width: int = 3,
    show_labels: bool = True,
    show_approved: bool = True,
    show_rejected: bool = True,
    show_undecided: bool = True,
) -> Image.Image:
    
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ann in anns:
        ann_id = str(ann["id"])
        x, y, w, h = ann["bbox"]
        status = decisions.get(ann_id, "undecided")

        # Color by status
        if not st.session_state.get("chk_random_colors", False):
            # Use a deterministic "random" color per annotation id so colors stay stable across rerenders
            color = {"approved": (0, 200, 0), "rejected": (220, 0, 0)}.get(status, (40, 120, 255))
        else:
            # Use a deterministic "random" color per annotation id so colors stay stable across rerenders
            try:
                seed = int(ann_id)
            except Exception:
                seed = hash(ann_id) & 0xFFFFFFFF
            rnd = random.Random(seed)
            color = (rnd.randint(40, 220), rnd.randint(40, 220), rnd.randint(40, 220))

        # Checks
        if status == "rejected" and not show_rejected:
            continue
        if status == "approved" and not show_approved:
            continue
        if status == "undecided" and not show_undecided:
            continue
        
        if st.session_state.get(f"chk_reveal_{ann_id}", True) is False:
            continue

        # Draw rectangle
        if st.session_state.get("chk_show_boxes", True):
            draw.rectangle([x, y, x + w, y + h], outline=color, width=line_width)

        elif show_labels:
            # Draw a diagonal line to indicate presence
            draw.line([x, y, x + w / 2.0, y + h / 2.0], fill=color, width=line_width)

        # Draw center marker
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r = 3

        draw.circle(
            [cx_i, cy_i],
            radius=r, 
            outline=(0,0,0), 
            # outline=(255,255,255), 
            fill=color
        )

        # Inner dot colored by status
        dot_r = max(1, line_width)
        draw.ellipse([cx_i - dot_r, cy_i - dot_r, cx_i + dot_r, cy_i + dot_r], fill=color)

        # Compose label text
        label = str(ann_id)
        maybe_score = ann.get("score", None)
        if maybe_score is not None:
            label += f" ({maybe_score:.2f})"

        # --- FIX: Pillow 10+ compatibility ---
        if font:
            try:
                # Pillow >=10: use textbbox
                bbox = draw.textbbox((x, y), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            except AttributeError:
                # Pillow <10 fallback
                tw, th = draw.textsize(label, font=font)

        else:
            tw, th = 0, 0
        # -------------------------------------

        # if status != "approved" and 
        if show_labels:
            pad = 2
            x_bg = max(0, x)
            y_bg = max(0, y - th - 2 * pad)
            bg = [
                x_bg, 
                y_bg, 
                x_bg + tw + 2 * pad, 
                y_bg + th + 2 * pad
            ]
            draw.rectangle(bg, fill=color)
            draw.text((x_bg + pad, max(0, y_bg + pad)), label, fill=(255, 255, 255), font=font)

    return img

def load_progress(progress_path: Path) -> Dict[int, str]:
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            return progress
        except Exception:
            print('Warning: Failed to load progress file; starting fresh.')
            pass
    return {}  # {ann_id: "approved"/"rejected"}

def save_progress(progress_path: Path, decisions: Dict[int, str]) -> None:
    progress_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

def export_filtered_coco(coco: Dict, decisions: Dict[int, str], out_path: Path) -> None:
    approved_ids = {int(ann_id) for ann_id, v in decisions.items() if v == "approved"}
    
    filtered_anns = [a for a in coco["annotations"] if a["id"] in approved_ids]
    # Keep only images that still have at least one approved annotation (or keep all; choose policy)
    image_ids_with_anns = {a["image_id"] for a in filtered_anns}
    filtered_images = [im for im in coco["images"] if im["id"] in image_ids_with_anns]
    filtered = {
        # "images": filtered_images,
        "images": coco["images"],  # keep all images even if some have no approved boxes (easier to review rejected/undecided later)
        "annotations": filtered_anns,
        "categories": coco["categories"],
    }
    out_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Bounding Box Approver", layout="wide")

with st.sidebar:
    st.header("Dataset Settings")
    images_dir = Path(st.text_input("Images directory", DEFAULT_IMAGES_DIR))
    coco_path = Path(st.text_input("COCO annotations path", DEFAULT_COCO_PATH))
    autosave_path = Path(st.text_input("Autosave progress file", AUTOSAVE_PATH))
    export_path = Path(st.text_input("Filtered COCO output", FILTERED_COCO_OUT))

    line_w = st.slider("Box line width", 0, 10, 2)

    st.toggle("Stretch images", key="chk_stretch_images", value=False)
    st.toggle("Show bounding boxes", key="chk_show_boxes", value=True)
    st.toggle("Show bounding box labels", key="chk_show_box_labels", value=True)
    st.toggle("Random colors", key="chk_random_colors", value=False)
    st.toggle("Show approved", key="chk_show_approved", value=True)
    st.toggle("Show rejected", key="chk_show_rejected", value=True)
    st.toggle("Show undecided", key="chk_show_undecided", value=True)
    st.toggle("Show only images with undecided boxes", key="chk_show_undecided_images", value=False)

    if st.button("Reload COCO"):
        st.session_state.pop("coco", None)
        st.session_state.pop("img_by_id", None)
        st.session_state.pop("anns_by_img", None)
        st.session_state.pop("cat_name", None)
        st.session_state.pop("image_ids", None)

# Load COCO once
if "coco" not in st.session_state:
    try:
        coco = load_coco(coco_path)
        img_by_id, anns_by_img, cat_name = index_coco(coco)
        image_ids = [im["id"] for im in coco["images"]]
        st.session_state.coco = coco
        st.session_state.img_by_id = img_by_id
        st.session_state.anns_by_img = anns_by_img
        st.session_state.cat_name = cat_name
        st.session_state.image_ids = image_ids

    except Exception as e:
        st.error(f"Failed to load COCO: {e}")
        st.stop()

coco = st.session_state.coco
img_by_id = st.session_state.img_by_id
anns_by_img = st.session_state.anns_by_img
cat_name = st.session_state.cat_name
image_ids = st.session_state.image_ids

# Progress (decisions)
if "decisions" not in st.session_state:
    st.session_state.decisions = load_progress(autosave_path)  # {ann_id: "approved"/"rejected"}

# Image index
if "idx" not in st.session_state:
    st.session_state.idx = 0

top_l, top_r = st.columns([3, 2])
with top_l:
    st.title("Bounding Box Approver")
    st.caption("Green=approved, Red=rejected, Blue=undecided. Your changes autosave.")

with top_r:
    c1, c2 = st.columns([1,1])
    with c1:
        st.metric("Total images", len(image_ids))
    with c2:
        total_anns = len(coco["annotations"])
        decided = sum(1 for v in st.session_state.decisions.values() if v in ("approved", "rejected"))
        st.metric("Boxes decided", f"{decided} / {total_anns} ({decided / total_anns * 100:.1f}%)")

# Controls
ctrl_l, ctrl_m, ctrl_r = st.columns([1, 1, 2])
with ctrl_l:
    if st.button("⟵ Previous", use_container_width=True, shortcut="a"):
    # if st_shortcuts.shortcut_button("⟵ Previous", ['arrowleft', 'a'], use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)

with ctrl_m:
    if st.button("Next ⟶", use_container_width=True, shortcut="d"):
    # if st_shortcuts.shortcut_button("Next ⟶", ['arrowright', 'd'], use_container_width=True):
        if st.session_state.chk_show_undecided_images:
            # Find next image with undecided boxes
            found = False
            for next_idx in range(st.session_state.idx + 1, len(image_ids)):
                img_id = image_ids[next_idx]
                anns = anns_by_img.get(img_id, [])
                for ann in anns:
                    ann_id = str(ann["id"])
                    current = st.session_state.decisions.get(ann_id, "undecided")
                    if current == "undecided":
                        st.session_state.idx = next_idx
                        found = True
                        break
                if found:
                    break
            if not found:
                # st.info("No more images with undecided boxes.")
                st.session_state.idx = 0

        else:
            st.session_state.idx = min(len(image_ids) - 1, st.session_state.idx + 1)

with ctrl_r:
    if st.button("Export Filtered COCO (Approved Only)", use_container_width=True):
        export_filtered_coco(coco, st.session_state.decisions, export_path)
        st.success(f"Exported to: {export_path.resolve()}")

st.slider(
    "Image",
    0, 
    len(image_ids) - 1, 
    # value=st.session_state.idx, 
    # key="slider_image_id", 
    key="idx", 
    # on_change=lambda: st.session_state.update(idx=st.session_state.slider_image_id)
)

# Current image
idx = st.session_state.idx
# idx = st.session_state.slider_image_id
img_id = image_ids[idx]
im_info = img_by_id[img_id]
img_file = images_dir / im_info["file_name"]

anns = anns_by_img.get(img_id, [])
decisions = st.session_state.decisions

# Set default values of the radio buttons
for ann in anns:
    ann_id = str(ann["id"])
    st.session_state.setdefault(f"chk_reveal_{ann_id}", True)
    st.session_state[f"_chk_reveal_{ann_id}"] = st.session_state[f"chk_reveal_{ann_id}"]

# Bulk actions
bulk_1, bulk_2, bulk_3, bulk_4 = st.columns([1,1,1,1])
with bulk_1:
    if st.button(
        "Approve ALL boxes in this image", 
        type="primary", 
        use_container_width=True, 
        shortcut="w", 
        icon=":material/check:"
    ):
        for ann in anns:
            decisions[str(ann["id"])] = "approved"
        
        save_progress(autosave_path, decisions)

with bulk_2:
    if st.button(
        "Reject ALL boxes in this image",
        shortcut="s",
        type="primary",
        use_container_width=True,
        icon=":material/cancel:"
    ):
        for ann in anns:
            decisions[str(ann["id"])] = "rejected"
        save_progress(autosave_path, decisions)

with bulk_3:
    if st.button("Reveal all boxes", use_container_width=True, icon=":material/visibility:"):
        for ann in anns:
            st.session_state[f"chk_reveal_{ann['id']}"] = True

with bulk_4:
    if st.button("Hide all boxes", use_container_width=True, icon=":material/visibility_off:"):
        for ann in anns:
            st.session_state[f"chk_reveal_{ann['id']}"] = False



# Get bbox stats=
bbox_stats = {"approved": 0, "rejected": 0, "undecided": 0}
for ann in anns:
    ann_id = str(ann["id"])
    current = decisions.get(ann_id, "undecided")
    bbox_stats[current] += 1

# 
def store_reveal_chenckbox_value(ann_id):
    st.session_state[f"chk_reveal_{ann_id}"] = st.session_state[f"_chk_reveal_{ann_id}"]

def store_radio_value(ann_id):
    decisions[ann_id] = st.session_state[f"_radio_{ann_id}"]

col1, col2 = st.columns([3,1])
with col2:
    st.write(
        f"Boxes for image #{idx} — {im_info.get('file_name', '')}"
    )
    st.write(
        f"Total: {len(anns):,d} (✅ {bbox_stats['approved']:,d} | ❌ {bbox_stats['rejected']:,d} | ❔ {bbox_stats['undecided']:,d})"
    )
    with st.container(height=700,):
        if not anns:
            st.error("No annotations for this image.")

        else:
            # Table-like controls
            for ann in sorted(anns, key=lambda a: a.get("score", 0.0), reverse=True):
                ann_id = str(ann["id"])
                current = decisions.get(ann_id, "undecided")
                
                c1, c2 = st.columns([4,1])
                with c1:
                    ann_key = f"_radio_{ann_id}"
                    st.session_state[ann_key] = current
                    choice = st.radio(
                            f"AnnID: {ann_id}",
                            options=["undecided", "approved", "rejected"],
                        horizontal=True,
                        key=ann_key,
                        on_change=store_radio_value,
                        args=(ann_id,)
                    )
                if choice != current:
                    decisions[ann_id] = choice
                    save_progress(autosave_path, decisions)

                with c2:
                    st.checkbox(
                        f"R", 
                        key=f"_chk_reveal_{ann_id}",
                        # value=True, 
                        disabled=False,
                        on_change=store_reveal_chenckbox_value,
                        args=(ann_id,)
                    )



with col1:
    preview = draw_boxes(
        img_file, 
        anns, 
        decisions, 
        cat_name, 
        line_width=line_w, 
        show_labels=st.session_state.chk_show_box_labels,
        show_undecided=st.session_state.chk_show_undecided,
        show_approved=st.session_state.chk_show_approved,
        show_rejected=st.session_state.chk_show_rejected
    )
    
    st.image(
        preview, 
        width='stretch' if st.session_state.chk_stretch_images else 'content',
        caption=im_info.get("file_name", "")
    )

# st.checkbox("Show image info (JSON)", key="chk_show_json")
# if st.session_state.chk_show_json:
#     st.json(im_info)

# Footer autosave notice
st.caption(f"Progress autosaved to: `{autosave_path}` — Export when ready to `{FILTERED_COCO_OUT}`.")

# st_shortcuts.add_shortcuts(
#     chk_show_box_labels='q',
#     chk_random_colors='r',
# )