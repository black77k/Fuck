"""
==============================================================
بوت تيليجرام تعليمي لكشف الأشخاص + رسم الهيكل العظمي (Pose)
==============================================================
الميزات:
  - كشف الأشخاص بدقة عالية باستخدام نموذج YOLO11-Pose.
  - رسم صندوق (Bounding Box) + هيكل عظمي (Skeleton) لكل شخص.
  - إضافة علامة مائية/حقوق (Watermark) شفافة احترافية على الصورة الناتجة.
  - دعم الصور، الفيديو، وملفات GIF (المتحركة) - مع تحسين أداء بسيط
    لتخطي بعض الإطارات في الفيديوهات الطويلة.
  - رسالة نتيجة احترافية باللغة الإنجليزية فقط، تنتهي بحقوق البوت.

الغرض: تعليمي بحت (مادة أمن سيبراني / برمجة) لشرح مبادئ Object
Detection و Pose Estimation والرؤية الحاسوبية.

المتطلبات: راجع requirements.txt
يتطلب أيضًا تثبيت "ffmpeg" كحزمة نظام (apt) على السيرفر لدعم الفيديو/GIF.

التشغيل:
  1) ضع التوكن في متغير البيئة TELEGRAM_BOT_TOKEN
  2) ضع صورة العلامة المائية (بخلفية شفافة PNG) في: assets/Copyright.png
  3) شغّل: python bot.py
==============================================================
"""

import os
import io
import logging
import subprocess
import tempfile
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
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

# نموذج Pose بدل نموذج الكشف العادي (يعطي Box + نقاط الهيكل العظمي معًا)
# الخيارات (من الأخف والأسرع إلى الأدق والأثقل):
#   yolo11n-pose.pt  -> سريع، دقة جيدة
#   yolo11s-pose.pt  -> متوازن (الافتراضي هنا)
#   yolo11m-pose.pt  -> دقة أعلى، أبطأ قليلاً
#   yolo11l-pose.pt / yolo11x-pose.pt -> أعلى دقة، يحتاج موارد أكبر
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo11s-pose.pt")

# حجم الصورة الداخلي أثناء الاستدلال - رفعه يحسّن الدقة (خصوصًا للأشخاص
# الصغار/البعيدين في الصورة) على حساب بعض السرعة
INFERENCE_IMG_SIZE = 1280

CONFIDENCE_THRESHOLD = 0.30

# مسار العلامة المائية (يجب أن تكون PNG بقناة شفافية Alpha)
WATERMARK_PATH = os.path.join(os.path.dirname(__file__), "assets", "Copyright.png")
# نسبة عرض العلامة المائية من عرض الصورة الأصلية
WATERMARK_WIDTH_RATIO = 0.18
WATERMARK_MARGIN_RATIO = 0.02
WATERMARK_OPACITY = 0.85  # 0.0 شفاف تمامًا -> 1.0 معتم تمامًا

# لتحسين أداء معالجة الفيديو: نعالج كل N إطار فقط ونكرر آخر رسم على البقية
VIDEO_FRAME_SKIP = 2
# حد أقصى لعدد الثواني المسموح معالجتها من أي فيديو (حماية من التعليق الطويل)
VIDEO_MAX_SECONDS = 20

BOX_COLOR = (0, 255, 60)
SKELETON_COLOR = (0, 200, 255)
JOINT_COLOR = (255, 80, 80)
TEXT_COLOR = (255, 255, 255)

# أزواج نقاط الهيكل العظمي حسب معيار COCO (17 نقطة)
SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
KEYPOINT_CONF_THRESHOLD = 0.5

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

logger.info("جاري تحميل نموذج YOLO11-Pose ... (%s)", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("تم تحميل النموذج بنجاح.")

# تحميل العلامة المائية مرة واحدة عند بدء التشغيل (إن وُجدت)
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

def run_pose_detection(image_bgr: np.ndarray):
    """يشغّل YOLO11-Pose على الصورة ويعيد نتائج الإطار الأول."""
    results = model.predict(
        source=image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=INFERENCE_IMG_SIZE,
        classes=[0],  # فئة "person" فقط
        verbose=False,
    )
    return results[0] if results else None


def draw_skeleton_overlay(image_bgr: np.ndarray, result) -> Tuple[np.ndarray, int]:
    """
    يرسم على الصورة:
      - صندوق حول كل شخص
      - هيكل عظمي (خطوط تصل بين نقاط الجسم) لكل شخص
      - نقاط المفاصل
    يعيد الصورة المعدّلة وعدد الأشخاص المكتشفين.
    """
    output = image_bgr.copy()
    if result is None or result.boxes is None:
        return output, 0

    boxes = result.boxes
    keypoints = result.keypoints  # قد تكون None إن لم يدعمها النموذج
    person_count = len(boxes)

    for idx in range(person_count):
        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 2)

        if keypoints is not None:
            kpts = keypoints.data[idx].cpu().numpy()  # shape: (17, 3) -> x, y, conf

            # رسم خطوط الهيكل العظمي
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

            # رسم نقاط المفاصل
            for (x, y, c) in kpts:
                if c > KEYPOINT_CONF_THRESHOLD:
                    cv2.circle(output, (int(x), int(y)), 3, JOINT_COLOR, -1)

    return output, person_count


def apply_watermark(image_bgr: np.ndarray) -> np.ndarray:
    """يدمج العلامة المائية بشفافية احترافية في الزاوية السفلية اليمنى."""
    if _watermark_rgba is None:
        return image_bgr

    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    img_w, img_h = base.size

    wm_w = max(1, int(img_w * WATERMARK_WIDTH_RATIO))
    wm_ratio = wm_w / _watermark_rgba.width
    wm_h = max(1, int(_watermark_rgba.height * wm_ratio))
    watermark = _watermark_rgba.resize((wm_w, wm_h), Image.LANCZOS)

    # ضبط الشفافية الإضافية دون المساس بالشفافية الأصلية للصورة
    if WATERMARK_OPACITY < 1.0:
        alpha = watermark.split()[3].point(lambda p: int(p * WATERMARK_OPACITY))
        watermark.putalpha(alpha)

    margin = int(img_w * WATERMARK_MARGIN_RATIO)
    pos = (img_w - wm_w - margin, img_h - wm_h - margin)

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(watermark, pos, watermark)
    combined = Image.alpha_composite(base, layer).convert("RGB")

    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


def process_frame(image_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
    """يجمع خطوات الكشف + الرسم + العلامة المائية لصورة واحدة."""
    result = run_pose_detection(image_bgr)
    annotated, count = draw_skeleton_overlay(image_bgr, result)
    annotated = apply_watermark(annotated)
    return annotated, count


def build_caption(person_count: int) -> str:
    """رسالة النتيجة الاحترافية بالإنجليزية فقط."""
    return (
        "✅ *Detection Complete*\n\n"
        f"👥 Persons Detected: *{person_count}*\n"
        "🧠 Model: YOLO11-Pose\n"
        "📐 Skeleton overlay applied\n\n"
        "🤖 Bot developed by @usta77k"
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

        annotated, count = process_frame(image_bgr)

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
    """يحوّل ملف الفيديو الناتج إلى صيغة متوافقة مع تيليجرام (H.264 أو GIF)."""
    if as_gif:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "fps=12,scale=480:-1:flags=lanczos",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)


async def handle_video_like(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يعالج الفيديو أو ملف GIF (Animation) إطارًا تلو الآخر."""
    message = update.message
    is_gif = message.animation is not None
    media = message.animation if is_gif else message.video

    processing_msg = await message.reply_text(
        "⏳ Processing video, this may take a while..."
    )

    tmp_in = tmp_out_raw = tmp_out_final = None
    cap = writer = None
    try:
        media_file = await media.get_file()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
            tmp_in = f_in.name
        await media_file.download_to_drive(tmp_in)

        cap = cv2.VideoCapture(tmp_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(fps * VIDEO_MAX_SECONDS)

        tmp_out_raw = tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_out_raw, fourcc, fps, (width, height))

        frame_idx = 0
        last_annotated = None
        max_persons_seen = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame_idx >= max_frames:
                break

            if frame_idx % VIDEO_FRAME_SKIP == 0 or last_annotated is None:
                annotated, count = process_frame(frame)
                last_annotated = annotated
                max_persons_seen = max(max_persons_seen, count)
            else:
                annotated = last_annotated

            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

        tmp_out_final = tempfile.NamedTemporaryFile(
            suffix=".gif" if is_gif else ".mp4", delete=False
        ).name
        _ffmpeg_convert(tmp_out_raw, tmp_out_final, as_gif=is_gif)

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
        "Send me a photo, video, or GIF containing people, and I will detect "
        "them using YOLO11-Pose and draw a skeleton overlay on each person.\n\n"
        "This is an educational bot for teaching Object Detection & "
        "Computer Vision.\n\n"
        "🤖 Bot developed by @usta77k"
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
