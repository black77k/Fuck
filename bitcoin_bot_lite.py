"""
==============================================================
بوت تيليجرام تعليمي لكشف الأشخاص (Person Detection Bot)
==============================================================
الفكرة:
  - المستخدم يرسل صورة للبوت.
  - البوت يستخدم نموذج YOLO11 لكشف الأشخاص (class: person) في الصورة.
  - يتم رسم مربع (Bounding Box) حول كل شخص، مع خط/نقطة دلالية أعلى
    المربع (تشبه أسلوب عرض "Tracking Overlay" المستخدم في شروحات
    الرؤية الحاسوبية)، ونسبة الثقة (Confidence) لكل كشف.
  - يتم إرسال الصورة المعدّلة + تقرير نصي (عدد الأشخاص، الثقة، ...)
    إلى المستخدم على تيليجرام.

الغرض: تعليمي بحت (مادة أمن سيبراني / برمجة) لشرح مبادئ:
  - Object Detection باستخدام الشبكات العصبية (YOLO)
  - معالجة الصور باستخدام OpenCV
  - بناء بوتات تفاعلية على تيليجرام تستقبل وترسل ملفات وسائط

المتطلبات: راجع requirements.txt
التشغيل:
  1) ضع التوكن الخاص بالبوت في متغير البيئة TELEGRAM_BOT_TOKEN
     أو مباشرة في متغير BOT_TOKEN أسفل الملف (غير موصى به للإنتاج).
  2) شغّل: python bot.py
==============================================================
"""

import os
import io
import logging
from typing import List, Tuple

import cv2
import numpy as np
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

# من الأفضل وضع التوكن في متغير بيئة بدلاً من كتابته مباشرة في الكود
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8789888929:AAE29y-R7t8ToIUiVxJ5Kw_SyzSbxt-aWoQ")

# اسم/مسار نموذج YOLO. "yolo11n.pt" هو النسخة الخفيفة (nano) وتُحمَّل تلقائيًا
# عند أول تشغيل من مستودع Ultralytics الرسمي. يمكنك استخدام yolo11s.pt أو
# yolo11m.pt لدقة أعلى مقابل سرعة أقل.
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo11n.pt")

# الحد الأدنى لثقة الكشف (Confidence Threshold) لعرض الصندوق
CONFIDENCE_THRESHOLD = 0.35

# في مجموعة بيانات COCO التي يتدرب عليها YOLO الافتراضي، الشخص = class 0
PERSON_CLASS_ID = 0

# ألوان الرسم (BGR لأن OpenCV يستخدم BGR وليس RGB)
BOX_COLOR = (0, 255, 60)       # أخضر
LINE_COLOR = (0, 200, 255)     # برتقالي/أصفر لخط التتبع
TEXT_COLOR = (255, 255, 255)   # أبيض

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# تحميل النموذج مرة واحدة عند بدء تشغيل البوت (وليس مع كل رسالة) لتوفير الوقت
logger.info("جاري تحميل نموذج YOLO11 ... (%s)", MODEL_PATH)
model = YOLO(MODEL_PATH)
logger.info("تم تحميل النموذج بنجاح.")


# --------------------------------------------------------------------------
# دوال المعالجة (Core Logic)
# --------------------------------------------------------------------------

def detect_persons(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    """
    يشغّل YOLO على الصورة ويعيد قائمة بمربعات الأشخاص المكتشفين فقط.

    Returns:
        قائمة من (x1, y1, x2, y2, confidence) لكل شخص تم كشفه.
    """
    results = model.predict(
        source=image_bgr,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],  # نطلب من YOLO كشف فئة "person" فقط
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append((int(x1), int(y1), int(x2), int(y2), conf))

    return detections


def draw_esp_style_overlay(
    image_bgr: np.ndarray,
    detections: List[Tuple[int, int, int, int, float]],
) -> np.ndarray:
    """
    يرسم على الصورة:
      - مربع (Bounding Box) حول كل شخص
      - خط عمودي قصير من أعلى منتصف المربع (خط تتبع/تحديد)
      - نص يوضّح رقم الشخص ونسبة الثقة
    """
    output = image_bgr.copy()

    for idx, (x1, y1, x2, y2, conf) in enumerate(detections, start=1):
        # 1) رسم المربع حول الشخص
        cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # 2) خط "تتبع" بسيط يمتد من أعلى الصورة إلى أعلى المربع
        #    (أسلوب شائع في عروض ESP لتوضيح موقع الهدف)
        center_x = (x1 + x2) // 2
        cv2.line(output, (center_x, 0), (center_x, y1), LINE_COLOR, 1)
        # نقطة صغيرة عند منتصف أعلى الرأس
        cv2.circle(output, (center_x, y1), 3, LINE_COLOR, -1)

        # 3) خلفية صغيرة للنص + النص نفسه (رقم الشخص + نسبة الثقة)
        label = f"Person {idx} ({conf * 100:.0f}%)"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            output,
            (x1, max(0, y1 - text_h - 8)),
            (x1 + text_w + 6, y1),
            BOX_COLOR,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 3, max(12, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # عداد إجمالي في أعلى يسار الصورة
    summary = f"Persons detected: {len(detections)}"
    cv2.putText(
        output, summary, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        output, summary, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA,
    )

    return output


# --------------------------------------------------------------------------
# معالجات تيليجرام (Telegram Handlers)
# --------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "أهلًا 👋\n"
        "أرسل لي أي صورة تحتوي على أشخاص، وسأقوم بتحليلها باستخدام "
        "YOLO11 وأعرض لك مواقعهم مع نسبة الثقة لكل كشف.\n\n"
        "هذا بوت تعليمي لشرح مبادئ Object Detection والرؤية الحاسوبية."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    processing_msg = await message.reply_text("⏳ جاري تحليل الصورة ...")

    try:
        # تيليجرام يرسل الصورة بعدة أحجام، نأخذ أعلى دقة متاحة
        photo_file = await message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # تحويل البايتات إلى صورة OpenCV (BGR)
        np_arr = np.frombuffer(photo_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            await processing_msg.edit_text("❌ تعذّر قراءة الصورة، حاول مرة أخرى.")
            return

        detections = detect_persons(image_bgr)
        annotated = draw_esp_style_overlay(image_bgr, detections)

        # ترميز الصورة الناتجة إلى JPEG في الذاكرة لإرسالها مباشرة
        success, encoded_img = cv2.imencode(".jpg", annotated)
        if not success:
            await processing_msg.edit_text("❌ حدث خطأ أثناء إنشاء الصورة الناتجة.")
            return

        output_buffer = io.BytesIO(encoded_img.tobytes())
        output_buffer.name = "detected.jpg"

        caption = (
            f"✅ تم الكشف عن {len(detections)} شخص/أشخاص.\n"
            + "\n".join(
                f"• شخص {i}: ثقة {c * 100:.1f}%"
                for i, (*_, c) in enumerate(detections, start=1)
            )
            if detections
            else "لم يتم العثور على أي شخص في الصورة."
        )

        await message.reply_photo(photo=output_buffer, caption=caption)
        await processing_msg.delete()

    except Exception as exc:  # noqa: BLE001
        logger.exception("خطأ أثناء معالجة الصورة")
        await processing_msg.edit_text(f"❌ حدث خطأ غير متوقع: {exc}")


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("من فضلك أرسل صورة (Photo) وليس نوع ملف آخر.")


# --------------------------------------------------------------------------
# نقطة التشغيل الرئيسية
# --------------------------------------------------------------------------

def main() -> None:
    if BOT_TOKEN == "PUT_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "الرجاء ضبط متغير البيئة TELEGRAM_BOT_TOKEN بتوكن البوت الخاص بك "
            "(احصل عليه من @BotFather في تيليجرام)."
        )

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(~filters.PHOTO & ~filters.COMMAND, handle_other))

    logger.info("البوت يعمل الآن ... اضغط Ctrl+C للإيقاف.")
    app.run_polling()


if __name__ == "__main__":
    main()
