"""
==============================================================
بوت تيليجرام تعليمي لكشف الأشخاص + رسم الهيكل العظمي (Pose)
==============================================================
الميزات:
  - كشف الأشخاص بدقة عالية باستخدام نموذج YOLO11-Pose (للصور).
  - نموذج أخف مخصص للفيديو/GIF لتفادي التعليق أو نفاد الذاكرة على
    سيرفرات محدودة الموارد (مثل خطط Railway المجانية/التجريبية).
  - رسم احترافي بأسلوب HUD: صندوق بزوايا (Corner Brackets) + هيكل
    عظمي للأشخاص الواضحين فقط.
  - إضافة علامة مائية/حقوق (Watermark) شفافة احترافية.
  - معالجة الفيديو تعمل في خيط منفصل (Thread) حتى لا يتجمد البوت،
    مع تحديث تقدم دوري للمستخدم وحد زمني أقصى صريح.
  - رسالة نتيجة احترافية باللغة الإنجليزية فقط، تنتهي بحقوق البوت.

الغرض: تعليمي بحت (مادة أمن سيبراني / برمجة) لشرح مبادئ Object
Detection و Pose Estimation والرؤية الحاسوبية.

المتطلبات: راجع requirements.txt
يتطلب أيضًا تثبيت "ffmpeg" كحزمة نظام (apt) على السيرفر لدعم الفيديو/GIF.

التشغيل:
  1) ضع التوكن في متغير البيئة TELEGRAM_BOT_TOKEN (أو داخل الكود أدناه)
  2) ضع صورة العلامة المائية (بخلفية شفافة PNG) في: assets/Copyright.png
  3) شغّل: python bot.py
==============================================================
"""

import os
import io
import time
import shutil
import asyncio
import logging
import subprocess
import tempfile
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --------------------------------------------------------------------------
# الإعدادات العامة (Configuration)
# --------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8700204975:AAF6P0jJuDdCuJFYlxC4f3LITpaFg6RpEg4")

# نموذج دقيق للصور الثابتة (جودة أعلى لأنها معالجة لمرة واحدة فقط)
PHOTO_MODEL_PATH = os.environ.get("YOLO_PHOTO_MODEL", "yolo11s-pose.pt")
PHOTO_INFERENCE_IMG_SIZE = 1280

# نموذج أخف مخصص للفيديو/GIF (كل إطار يُعاد معالجته بالكامل، لذا يجب أن
# يكون أخف بكثير حتى لا يتجمد السيرفر أو ينفد رامه)
VIDEO_MODEL_PATH = os.environ.get("YOLO_VIDEO_MODEL", "yolo11n-pose.pt")
VIDEO_INFERENCE_IMG_SIZE = 480

CONFIDENCE_THRESHOLD = 0.30

# مسار العلامة المائية (يجب أن تكون PNG بقناة شفافية Alpha)
WATERMARK_PATH = os.path.join(os.path.dirname(__file__), "assets", "Copyright.png")
WATERMARK_WIDTH_RATIO = 0.18
WATERMARK_MARGIN_RATIO = 0.02
WATERMARK_OPACITY = 0.85

# --- إعدادات أداء الفيديو ---
# نعالج إطارًا واحدًا من كل N إطار فقط ونكرر آخر رسم على البقية
VIDEO_FRAME_SKIP = 4
# أقصى عدد ثوانٍ من الفيديو الأصلي يتم معالجتها (حماية من فيديوهات طويلة)
VIDEO_MAX_SECONDS = 8
# أقصى وقت معالجة فعلي بالثواني قبل إلغاء العملية بالكامل برسالة واضحة
VIDEO_PROCESSING_TIMEOUT = 90
# كل كم ثانية نُحدّث رسالة "جاري المعالجة" بنسبة التقدم
PROGRESS_UPDATE_INTERVAL = 4

# --- ألوان وأسلوب الرسم الاحترافي (HUD Style) ---
BOX_COLOR = (0, 255, 120)
SKELETON_COLOR = (0, 210, 255)
JOINT_COLOR = (255, 90, 90)
CORNER_LEN_RATIO = 0.22   # طول زاوية الصندوق كنسبة من أصغر ضلع
CORNER_THICKNESS = 3

# أزواج نقاط الهيكل العظمي حسب معيار COCO (17 نقطة)
SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KEYPOINT_CONF_THRESHOLD = 0.65

# لا نرسم الهيكل العظمي الكامل إلا للأشخاص الذين تشغل صورتهم نسبة كافية
# من الإطار (قريبون/واضحون بما يكفي لتقدير وضعية موثوق). البقية تُرسم
# لهم زوايا الصندوق فقط لتفادي هيكل متشابك غير دقيق.
MIN_PERSON_AREA_RATIO_FOR_SKELETON = 0.015  # 1.5% من مساحة الصورة

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logger.info("جاري تحميل نموذج الصور YOLO11-Pose ... (%s)", PHOTO_MODEL_PATH)
photo_model = YOLO(PHOTO_MODEL_PATH)
logger.info("جاري تحميل نموذج الفيديو الخفيف ... (%s)", VIDEO_MODEL_PATH)
video_model = YOLO(VIDEO_MODEL_PATH)
logger.info("تم تحميل النماذج بنجاح.")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
if not FFMPEG_AVAILABLE:
    logger.warning(
        "لم يتم العثور على ffmpeg على السيرفر! معالجة الفيديو/GIF لن تعمل "
        "حتى تُثبَّت حزمة ffmpeg."
    )

_watermark_rgba: Optional[Image.Image] = None
if os.path.exists(WATERMARK_PATH):
    _watermark_rgba = Image.open(WATERMARK_PATH).convert("RGBA")
    logger.info("تم تحميل العلامة المائية من: %s", WATERMARK_PATH)
else:
    logger.warning(
        "لم يتم العثور على العلامة المائية في %s - سيتم تجاهلها.", WATERMARK_PATH
    )


# --------------------------------------------------------------------------
# دوال الكشف والرسم (Core Logic)
# --------------------------------------------------------------------------

def run_pose_detection(model: YOLO, image_bgr: np.ndarray, imgsz: int):
    results = model.predict(
        source=image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=imgsz,
        classes=[0],  # فئة "person" فقط
        verbose=False,
    )
    return results[0] if results else None


def _draw_corner_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """يرسم صندوقًا بأسلوب زوايا (HUD) بدل مربع كامل - شكل احترافي أنظف."""
    w, h = x2 - x1, y2 - y1
    clen = max(6, int(min(w, h) * CORNER_LEN_RATIO))
    color = BOX_COLOR
    t = CORNER_THICKNESS

    # الزاوية العلوية اليسرى
    cv2.line(img, (x1, y1), (x1 + clen, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + clen), color, t, cv2.LINE_AA)
    # العلوية اليمنى
    cv2.line(img, (x2, y1), (x2 - clen, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + clen), color, t, cv2.LINE_AA)
    # السفلية اليسرى
    cv2.line(img, (x1, y2), (x1 + clen, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - clen), color, t, cv2.LINE_AA)
    # السفلية اليمنى
    cv2.line(img, (x2, y2), (x2 - clen, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - clen), color, t, cv2.LINE_AA)


def draw_skeleton_overlay(image_bgr: np.ndarray, result) -> Tuple[np.ndarray, int]:
    output = image_bgr.copy()
    if result is None or result.boxes is None:
        return output, 0

    boxes = result.boxes
    keypoints = result.keypoints
    person_count = len(boxes)

    img_h, img_w = output.shape[:2]
    img_area = float(img_w * img_h)

    for idx in range(person_count):
        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _draw_corner_box(output, x1, y1, x2, y2)

        box_area_ratio = ((x2 - x1) * (y2 - y1)) / img_area if img_area else 0
        draw_full_skeleton = box_area_ratio >= MIN_PERSON_AREA_RATIO_FOR_SKELETON

        if keypoints is not None and draw_full_skeleton:
            kpts = keypoints.data[idx].cpu().numpy()

            for (a, b) in SKELETON_PAIRS:
                if a >= len(kpts) or b >= len(kpts):
                    continue
                xa, ya, ca = kpts[a]
                xb, yb, cb = kpts[b]
                if ca > KEYPOINT_CONF_THRESHOLD and cb > KEYPOINT_CONF_THRESHOLD:
                    cv2.line(
                        output, (int(xa), int(ya)), (int(xb), int(yb)),
                        SKELETON_COLOR, 2, cv2.LINE_AA,
                    )

            for (x, y, c) in kpts:
                if c > KEYPOINT_CONF_THRESHOLD:
                    cv2.circle(output, (int(x), int(y)), 3, JOINT_COLOR, -1)

    return output, person_count


def apply_watermark(image_bgr: np.ndarray) -> np.ndarray:
    if _watermark_rgba is None:
        return image_bgr

    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    img_w, img_h = base.size

    wm_w = max(1, int(img_w * WATERMARK_WIDTH_RATIO))
    wm_ratio = wm_w / _watermark_rgba.width
    wm_h = max(1, int(_watermark_rgba.height * wm_ratio))
    watermark = _watermark_rgba.resize((wm_w, wm_h), Image.LANCZOS)

    if WATERMARK_OPACITY < 1.0:
        alpha = watermark.split()[3].point(lambda p: int(p * WATERMARK_OPACITY))
        watermark.putalpha(alpha)

    margin = int(img_w * WATERMARK_MARGIN_RATIO)
    pos = (img_w - wm_w - margin, img_h - wm_h - margin)

    pad = int(wm_w * 0.08)
    backdrop_box = (
        pos[0] - pad, pos[1] - pad,
        pos[0] + wm_w + pad, pos[1] + wm_h + pad,
    )
    backdrop = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(backdrop).rounded_rectangle(backdrop_box, radius=pad, fill=(0, 0, 0, 110))

    layer = Image.alpha_composite(Image.new("RGBA", base.size, (0, 0, 0, 0)), backdrop)
    layer.paste(watermark, pos, watermark)
    combined = Image.alpha_composite(base, layer).convert("RGB")

    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def process_frame(image_bgr: np.ndarray, *, for_video: bool = False) -> Tuple[np.ndarray, int]:
    model = video_model if for_video else photo_model
    imgsz = VIDEO_INFERENCE_IMG_SIZE if for_video else PHOTO_INFERENCE_IMG_SIZE
    result = run_pose_detection(model, image_bgr, imgsz)
    annotated, count = draw_skeleton_overlay(image_bgr, result)
    annotated = apply_watermark(annotated)
    return annotated, count


def build_caption(person_count: int) -> str:
    return (
        "✅ *Detection Complete*\n\n"
        f"👥 Persons Detected: *{person_count}*\n"
        "🧠 Model: YOLO11-Pose\n"
        "📐 Skeleton overlay applied\n\n"
        "🤖 Bot developed by @Berkocan77k"
    )


# --------------------------------------------------------------------------
# معالجة الصور (Photo Handler)
# --------------------------------------------------------------------------

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    processing_msg = await message.reply_text("⏳ Analyzing image...")

    try:
        photo_file = await message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        np_arr = np.frombuffer(photo_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            await processing_msg.edit_text("❌ Could not read the image. Please try again.")
            return

        loop = asyncio.get_running_loop()
        annotated, count = await loop.run_in_executor(
            None, lambda: process_frame(image_bgr, for_video=False)
        )

        success, encoded_img = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            await processing_msg.edit_text("❌ Failed to generate the output image.")
            return

        output_buffer = io.BytesIO(encoded_img.tobytes())
        output_buffer.name = "detected.jpg"

        await message.reply_photo(
            photo=output_buffer,
            caption=build_caption(count),
            parse_mode="Markdown",
        )
        await processing_msg.delete()

    except Exception as exc:  # noqa: BLE001
        logger.exception("خطأ أثناء معالجة الصورة")
        await processing_msg.edit_text(f"❌ Unexpected error: {exc}")


# --------------------------------------------------------------------------
# معالجة الفيديو / GIF (Video & Animation Handler)
# --------------------------------------------------------------------------

def _ffmpeg_convert(input_path: str, output_path: str, as_gif: bool) -> None:
    if as_gif:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "fps=10,scale=400:-1:flags=lanczos",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _process_video_sync(tmp_in: str, tmp_out_raw: str, progress_cb) -> int:
    """
    يعالج الفيديو إطارًا تلو الآخر (تعمل في Thread منفصل حتى لا تحجب البوت).
    progress_cb(processed_frames, total_frames) تُستدعى دوريًا للتحديث.
    """
    cap = cv2.VideoCapture(tmp_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * VIDEO_MAX_SECONDS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out_raw, fourcc, fps, (width, height))

    frame_idx = 0
    last_annotated = None
    max_persons_seen = 0
    last_progress_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= max_frames:
            break

        if frame_idx % VIDEO_FRAME_SKIP == 0 or last_annotated is None:
            annotated, count = process_frame(frame, for_video=True)
            last_annotated = annotated
            max_persons_seen = max(max_persons_seen, count)
        else:
            annotated = last_annotated

        writer.write(annotated)
        frame_idx += 1

        if time.time() - last_progress_time > PROGRESS_UPDATE_INTERVAL:
            progress_cb(frame_idx, max_frames)
            last_progress_time = time.time()

    cap.release()
    writer.release()
    return max_persons_seen


async def handle_video_like(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message

    if not FFMPEG_AVAILABLE:
        await message.reply_text(
            "❌ Video processing is unavailable: ffmpeg is not installed on "
            "the server. Please add 'ffmpeg' to the server's apt packages."
        )
        return

    is_gif = message.animation is not None
    media = message.animation if is_gif else message.video

    processing_msg = await message.reply_text("⏳ Processing video (0%)...")

    tmp_in = tmp_out_raw = tmp_out_final = None
    try:
        media_file = await media.get_file()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
            tmp_in = f_in.name
        await media_file.download_to_drive(tmp_in)

        tmp_out_raw = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False).name

        loop = asyncio.get_running_loop()

        def sync_progress(done: int, total: int) -> None:
            pct = int(min(done / max(total, 1), 1.0) * 100)
            asyncio.run_coroutine_threadsafe(
                processing_msg.edit_text(f"⏳ Processing video ({pct}%)..."),
                loop,
            )

        max_persons_seen = await asyncio.wait_for(
            loop.run_in_executor(
                None, _process_video_sync, tmp_in, tmp_out_raw, sync_progress
            ),
            timeout=VIDEO_PROCESSING_TIMEOUT,
        )

        await processing_msg.edit_text("⏳ Finalizing output...")

        tmp_out_final = tempfile.NamedTemporaryFile(
            suffix=".gif" if is_gif else ".mp4", delete=False
        ).name
        await loop.run_in_executor(
            None, _ffmpeg_convert, tmp_out_raw, tmp_out_final, is_gif
        )

        caption = build_caption(max_persons_seen)
        with open(tmp_out_final, "rb") as out_f:
            if is_gif:
                await message.reply_animation(
                    animation=out_f, caption=caption, parse_mode="Markdown"
                )
            else:
                await message.reply_video(
                    video=out_f, caption=caption, parse_mode="Markdown"
                )

        await processing_msg.delete()

    except asyncio.TimeoutError:
        await processing_msg.edit_text(
            "❌ Processing took too long and was cancelled. Try a shorter "
            "clip or a smaller resolution video."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("خطأ أثناء معالجة الفيديو")
        await processing_msg.edit_text(f"❌ Unexpected error while processing video: {exc}")

    finally:
        for path in (tmp_in, tmp_out_raw, tmp_out_final):
            if path and os.path.exists(path):
                os.remove(path)


# --------------------------------------------------------------------------
# أوامر عامة
# --------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Welcome!\n"
        "Send me a photo, short video, or GIF containing people, and I will "
        "detect them using YOLO11-Pose and draw a professional skeleton "
        "overlay on each person.\n\n"
        f"⏱ Videos/GIFs are limited to the first {VIDEO_MAX_SECONDS} seconds "
        "for performance reasons.\n\n"
        "This is an educational bot for teaching Object Detection & "
        "Computer Vision.\n\n"
        "🤖 Bot developed by @Berkocan77k"
    )


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Please send a photo, video, or GIF.")


# --------------------------------------------------------------------------
# نقطة التشغيل الرئيسية
# --------------------------------------------------------------------------

def main() -> None:
    if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "الرجاء ضبط متغير البيئة TELEGRAM_BOT_TOKEN بتوكن البوت الخاص بك."
        )

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video_like))
    app.add_handler(MessageHandler(filters.ANIMATION, handle_video_like))
    app.add_handler(
        MessageHandler(
            ~filters.PHOTO & ~filters.VIDEO & ~filters.ANIMATION & ~filters.COMMAND,
            handle_other,
        )
    )

    logger.info("البوت يعمل الآن ... اضغط Ctrl+C للإيقاف.")
    app.run_polling()


if __name__ == "__main__":
    main()
