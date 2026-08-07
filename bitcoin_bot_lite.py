"""
بوت تلجرام لمتابعة سعر البتكوين وتحليله الفني — نسخة خفيفة بدون pandas/numpy
=============================================================================
هذي النسخة مصممة للعمل على Pydroid3 / أندرويد بدون الحاجة لتثبيت numpy أو
pandas (اللي غالبًا يفشلوا بالتثبيت على الموبايل لأنه ما فيه wheel جاهز
ويحاول يبني من المصدر).

المكتبات المطلوبة فقط:
    pip install python-telegram-bot==21.4 requests

طريقة التشغيل:
    1. أنشئ بوت جديد عبر @BotFather في تلجرام واحصل على التوكن.
    2. ضع التوكن في متغير البيئة TELEGRAM_BOT_TOKEN أو مباشرة بالمتغير BOT_TOKEN أدناه.
    3. شغّل: python bitcoin_bot_lite.py
    4. بعد التأكد إنه شغّال، انقله لسيرفر (VPS) وشغّله عبر systemd أو tmux/screen
       ليعمل 24/7.

تنويه: كل تحليل يعطيه البوت هو مؤشر فني آلي مبني على بيانات تاريخية
(RSI و المتوسطات المتحركة و MACD)، وليس توصية استثمارية. الأسواق
متقلبة والقرار المالي مسؤوليتك الشخصية.
"""

import os
import logging
from datetime import datetime

import requests

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ------------------------------------------------------------------
# الإعدادات
# ------------------------------------------------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8789888929:AAE29y-R7t8ToIUiVxJ5Kw_SyzSbxt-aWoQ")
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_OHLC_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

AUTO_UPDATE_INTERVAL = 15 * 60  # كل 15 دقيقة

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# جلب البيانات
# ------------------------------------------------------------------
def get_current_price():
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
    }
    resp = requests.get(COINGECKO_PRICE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()["bitcoin"]
    return data["usd"], data["usd_24h_change"]


def get_price_history(days=30):
    """يرجّع لستة بسيطة من الأسعار (float) بدون pandas."""
    params = {"vs_currency": "usd", "days": days}
    resp = requests.get(COINGECKO_OHLC_URL, params=params, timeout=10)
    resp.raise_for_status()
    prices_raw = resp.json()["prices"]  # [[timestamp, price], ...]
    return [p[1] for p in prices_raw]


# ------------------------------------------------------------------
# التحليل الفني — بـ Python عادي فقط (lists)
# ------------------------------------------------------------------
def sma(values, period):
    """متوسط متحرك بسيط لآخر period قيمة."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values, period):
    """يرجّع لستة كاملة من EMA (متوسط أسي) بنفس طول values."""
    k = 2 / (period + 1)
    ema_vals = [values[0]]
    for price in values[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


def compute_rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(values):
    ema12 = ema_series(values, 12)
    ema26 = ema_series(values, 26)
    macd_line_series = [a - b for a, b in zip(ema12, ema26)]
    signal_series = ema_series(macd_line_series, 9)
    return macd_line_series[-1], signal_series[-1]


def analyze_market():
    prices = get_price_history(days=30)
    current_price = prices[-1]

    rsi = compute_rsi(prices)
    macd_line, signal_line = compute_macd(prices)
    sma7 = sma(prices, 7)
    sma25 = sma(prices, 25)

    score = 0
    reasons = []

    if rsi is not None:
        if rsi < 30:
            score += 1
            reasons.append(f"RSI منخفض ({rsi:.1f}) — منطقة تشبع بيعي محتملة")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI مرتفع ({rsi:.1f}) — منطقة تشبع شرائي محتملة")
        else:
            reasons.append(f"RSI محايد ({rsi:.1f})")

    if macd_line > signal_line:
        score += 1
        reasons.append("خط MACD فوق خط الإشارة — زخم إيجابي")
    else:
        score -= 1
        reasons.append("خط MACD تحت خط الإشارة — زخم سلبي")

    if sma7 and sma25:
        if sma7 > sma25:
            score += 1
            reasons.append("المتوسط المتحرك القصير فوق الطويل — اتجاه صاعد قصير المدى")
        else:
            score -= 1
            reasons.append("المتوسط المتحرك القصير تحت الطويل — اتجاه هابط قصير المدى")

    if score >= 2:
        signal = "🟢 ميل عام نحو الشراء (Bullish)"
    elif score <= -2:
        signal = "🔴 ميل عام نحو البيع (Bearish)"
    else:
        signal = "🟡 السوق محايد / غير واضح الاتجاه"

    text = (
        f"📊 *تحليل فني للبتكوين*\n"
        f"السعر الحالي: ${current_price:,.2f}\n\n"
        + "\n".join(f"• {r}" for r in reasons)
        + f"\n\n{signal}\n\n"
        f"⚠️ هذا تحليل آلي مبني على مؤشرات تاريخية فقط، وليس توصية "
        f"استثمارية أو مالية. تحقق دائمًا من مصادر إضافية قبل أي قرار."
    )
    return text


# ------------------------------------------------------------------
# أوامر البوت
# ------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "أهلاً 👋\n\n"
        "أنا بوت متابعة البتكوين. الأوامر المتاحة:\n"
        "/price — السعر الحالي\n"
        "/analyze — تحليل فني وإشارة عامة\n"
        "/subscribe — تفعيل التحديثات الدورية في هذه المحادثة\n"
        "/unsubscribe — إيقاف التحديثات الدورية\n"
    )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        price, change_24h = get_current_price()
        arrow = "🔺" if change_24h >= 0 else "🔻"
        await update.message.reply_text(
            f"💰 سعر البتكوين الحالي: *${price:,.2f}*\n"
            f"{arrow} التغيّر خلال 24 ساعة: {change_24h:.2f}%",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.exception("price_command failed")
        await update.message.reply_text(f"صار خطأ أثناء جلب السعر: {e}")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = analyze_market()
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.exception("analyze_command failed")
        await update.message.reply_text(f"صار خطأ أثناء التحليل: {e}")


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job_name = f"auto_update_{chat_id}"

    existing = context.job_queue.get_jobs_by_name(job_name)
    if existing:
        await update.message.reply_text("أنت مشترك أصلاً في التحديثات ✅")
        return

    context.job_queue.run_repeating(
        send_auto_update,
        interval=AUTO_UPDATE_INTERVAL,
        first=5,
        chat_id=chat_id,
        name=job_name,
    )
    await update.message.reply_text(
        f"تم تفعيل التحديثات كل {AUTO_UPDATE_INTERVAL // 60} دقيقة ✅"
    )


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job_name = f"auto_update_{chat_id}"
    jobs = context.job_queue.get_jobs_by_name(job_name)
    if not jobs:
        await update.message.reply_text("ما عندك اشتراك مفعّل أصلاً.")
        return
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("تم إيقاف التحديثات الدورية ❌")


async def send_auto_update(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    try:
        text = analyze_market()
        await context.bot.send_message(chat_id=job.chat_id, text=text, parse_mode="Markdown")
    except Exception:
        logger.exception("send_auto_update failed for chat_id=%s", job.chat_id)


# ------------------------------------------------------------------
# نقطة التشغيل
# ------------------------------------------------------------------
def main():
    if BOT_TOKEN == "ضع_التوكن_هنا":
        raise SystemExit(
            "ضع توكن البوت في متغير البيئة TELEGRAM_BOT_TOKEN أو مباشرة بالكود قبل التشغيل."
        )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("subscribe", subscribe_command))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_command))

    logger.info("البوت بدأ العمل عند %s", datetime.now())
    app.run_polling()


if __name__ == "__main__":
    main()
