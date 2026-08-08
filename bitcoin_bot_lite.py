"""
╔══════════════════════════════════════════════════════════════════╗
║        BITCOIN INTELLIGENCE BOT — ULTRA PROFESSIONAL EDITION      ║
╚══════════════════════════════════════════════════════════════════╝

هذي النسخة مصممة للعمل على سيرفر حقيقي (Railway وغيره) — تستخدم أقوى
المكتبات المتاحة: pandas, numpy, ta (Technical Analysis), ccxt (بيانات
مباشرة من Binance)، و mplfinance لرسم شموع يابانية حقيقية كصورة.

⚠️ لا تجرّبها على Pydroid3 / أندرويد — numpy/pandas ما تتثبت على
الموبايل (شفنا هذا سابقًا). هذي فقط للسيرفر.

المميزات:
  • بيانات شموع (OHLCV) مباشرة من Binance عبر ccxt — دقة وسرعة أعلى من CoinGecko
  • مؤشرات فنية احترافية عبر مكتبة ta: RSI, MACD, EMA20/50/200,
    Bollinger Bands, Stochastic RSI, ADX (قوة الاتجاه)
  • رسم بياني حقيقي بالشموع اليابانية + المتوسطات + بولينجر، يُرسل كصورة
  • نظام تسجيل نقاط موزون (Weighted Scoring) لإشارة نهائية
  • تنبيهات سعرية، اشتراكات دورية، تخزين JSON دائم — نفس فلسفة النسخة السابقة
  • أزرار تفاعلية (Inline Keyboard)

المكتبات المطلوبة (requirements.txt):
    python-telegram-bot==21.4
    requests
    pandas
    numpy
    ta
    ccxt
    mplfinance
    matplotlib

التشغيل:
    export TELEGRAM_BOT_TOKEN=توكنك
    python bitcoin_bot_ultra.py
"""

import os
import json
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import ta
import ccxt
import matplotlib
matplotlib.use("Agg")  # بدون واجهة رسومية — مهم على السيرفر
import mplfinance as mpf

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ══════════════════════════════════════════════════════════════════
#  الإعدادات العامة
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
#  ⬇️⬇️⬇️  ضع توكن البوت هنا فقط (سطر واحد) — أو اتركه فاضي واستخدم
#  متغير البيئة TELEGRAM_BOT_TOKEN من Railway (الطريقة الأأمن) ⬇️⬇️⬇️
# ══════════════════════════════════════════════════════════════════
BOT_TOKEN_MANUAL = "8789888929:AAE29y-R7t8ToIUiVxJ5Kw_SyzSbxt-aWoQ"   # مثال: "8789888929:AAE29y-R7t8ToIUiVxJ5Kw_SyzSbxt-aWoQ"

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or BOT_TOKEN_MANUAL

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"          # الشمعة الواحدة = ساعة
OHLCV_LIMIT = 200          # عدد الشموع المجلوبة (يكفي لـ EMA200)

DATA_FILE = "bot_data.json"
CACHE_TTL_SECONDS = 45
DEFAULT_AUTO_INTERVAL = 15 * 60
ALERT_CHECK_INTERVAL = 60
CHART_PATH = "chart.png"

logging.basicConfig(
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("BTC-ULTRA-BOT")

exchange = ccxt.binance({"enableRateLimit": True})


# ══════════════════════════════════════════════════════════════════
#  التخزين (JSON)
# ══════════════════════════════════════════════════════════════════
class Storage:
    def __init__(self, path: str):
        self.path = path
        self.data = {"subscriptions": {}, "alerts": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                logger.warning("تعذّر قراءة ملف التخزين، بدء بيانات جديدة.")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_subscription(self, chat_id: int, interval: int):
        self.data["subscriptions"][str(chat_id)] = interval
        self.save()

    def remove_subscription(self, chat_id: int):
        self.data["subscriptions"].pop(str(chat_id), None)
        self.save()

    def all_subscriptions(self):
        return dict(self.data["subscriptions"])

    def add_alert(self, chat_id: int, target_price: float, direction: str):
        alerts = self.data["alerts"].setdefault(str(chat_id), [])
        alerts.append({"price": target_price, "direction": direction, "triggered": False})
        self.save()

    def get_alerts(self, chat_id: int):
        return self.data["alerts"].get(str(chat_id), [])

    def clear_alerts(self, chat_id: int):
        self.data["alerts"][str(chat_id)] = []
        self.save()

    def all_alerts(self):
        return self.data["alerts"]

    def mark_triggered(self, chat_id: int, index: int):
        try:
            self.data["alerts"][str(chat_id)][index]["triggered"] = True
            self.save()
        except (KeyError, IndexError):
            pass


storage = Storage(DATA_FILE)


# ══════════════════════════════════════════════════════════════════
#  الكاش
# ══════════════════════════════════════════════════════════════════
@dataclass
class _Cache:
    value: Optional[object] = None
    timestamp: float = 0.0

    def is_fresh(self, ttl=CACHE_TTL_SECONDS):
        return self.value is not None and (time.time() - self.timestamp) < ttl


_ohlcv_cache = _Cache()


# ══════════════════════════════════════════════════════════════════
#  جلب البيانات من Binance عبر ccxt
# ══════════════════════════════════════════════════════════════════
def fetch_ohlcv_df() -> pd.DataFrame:
    """يرجّع DataFrame فيه Open/High/Low/Close/Volume آخر OHLCV_LIMIT شمعة."""
    if _ohlcv_cache.is_fresh():
        return _ohlcv_cache.value.copy()

    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    _ohlcv_cache.value = df
    _ohlcv_cache.timestamp = time.time()
    return df.copy()


def fetch_ticker_snapshot():
    """سعر لحظي + تغيّر 24 ساعة + أعلى/أدنى من Binance."""
    t = exchange.fetch_ticker(SYMBOL)
    return {
        "price": t["last"],
        "change_24h": t["percentage"],
        "high_24h": t["high"],
        "low_24h": t["low"],
        "volume_24h": t["quoteVolume"],
    }


# ══════════════════════════════════════════════════════════════════
#  المؤشرات الفنية عبر مكتبة ta (احترافية وموثوقة)
# ══════════════════════════════════════════════════════════════════
def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], window=min(200, len(df) - 1)).ema_indicator()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"])
    df["stoch_rsi_k"] = stoch_rsi.stochrsi_k() * 100
    df["stoch_rsi_d"] = stoch_rsi.stochrsi_d() * 100

    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()

    return df


# ══════════════════════════════════════════════════════════════════
#  نظام التسجيل الموزون (Weighted Scoring)
# ══════════════════════════════════════════════════════════════════
def build_full_analysis():
    df = fetch_ohlcv_df()
    df = enrich_with_indicators(df)
    last = df.iloc[-1]
    ticker = fetch_ticker_snapshot()

    score = 0.0
    signals = []

    # RSI (وزن 2)
    rsi = last["rsi"]
    if rsi < 30:
        score += 2
        signals.append(f"🟢 RSI = {rsi:.1f} → تشبع بيعي")
    elif rsi > 70:
        score -= 2
        signals.append(f"🔴 RSI = {rsi:.1f} → تشبع شرائي")
    else:
        signals.append(f"🟡 RSI = {rsi:.1f} → محايد")

    # MACD Histogram (وزن 1.5)
    if last["macd_hist"] > 0:
        score += 1.5
        signals.append(f"🟢 MACD Histogram = {last['macd_hist']:+.2f} → زخم صاعد")
    else:
        score -= 1.5
        signals.append(f"🔴 MACD Histogram = {last['macd_hist']:+.2f} → زخم هابط")

    # EMA20/50 Cross (وزن 1)
    if last["ema20"] > last["ema50"]:
        score += 1
        signals.append("🟢 EMA20 فوق EMA50 → اتجاه صاعد قصير المدى")
    else:
        score -= 1
        signals.append("🔴 EMA20 تحت EMA50 → اتجاه هابط قصير المدى")

    # EMA200 (اتجاه عام، وزن 1.5)
    if not pd.isna(last["ema200"]):
        if last["close"] > last["ema200"]:
            score += 1.5
            signals.append("🟢 السعر فوق EMA200 → اتجاه عام صاعد")
        else:
            score -= 1.5
            signals.append("🔴 السعر تحت EMA200 → اتجاه عام هابط")

    # Bollinger Bands (وزن 1)
    if last["close"] >= last["bb_upper"]:
        score -= 1
        signals.append("🔴 السعر عند الحد العلوي لبولينجر → احتمال تصحيح")
    elif last["close"] <= last["bb_lower"]:
        score += 1
        signals.append("🟢 السعر عند الحد السفلي لبولينجر → احتمال ارتداد")
    else:
        signals.append("🟡 السعر داخل نطاق بولينجر الطبيعي")

    # Stochastic RSI (وزن 1)
    k = last["stoch_rsi_k"]
    if not pd.isna(k):
        if k < 20:
            score += 1
            signals.append(f"🟢 Stochastic RSI = {k:.1f} → تشبع بيعي")
        elif k > 80:
            score -= 1
            signals.append(f"🔴 Stochastic RSI = {k:.1f} → تشبع شرائي")

    # ADX — قوة الاتجاه (لا يضيف للسكور، بس معلومة مهمة)
    adx = last["adx"]
    if not pd.isna(adx):
        strength = "قوي جدًا" if adx > 40 else "قوي" if adx > 25 else "ضعيف / تذبذب"
        signals.append(f"ℹ️ ADX = {adx:.1f} → قوة الاتجاه: {strength}")

    if score >= 4:
        verdict = "🟢🟢🟢 إشارة صعودية قوية جدًا (Very Strong Bullish)"
    elif score >= 2:
        verdict = "🟢 إشارة صعودية (Bullish)"
    elif score <= -4:
        verdict = "🔴🔴🔴 إشارة هبوطية قوية جدًا (Very Strong Bearish)"
    elif score <= -2:
        verdict = "🔴 إشارة هبوطية (Bearish)"
    else:
        verdict = "🟡 السوق متذبذب / بدون اتجاه واضح (Neutral)"

    return {
        "df": df,
        "ticker": ticker,
        "signals": signals,
        "verdict": verdict,
        "score": score,
    }


def format_analysis_message(result):
    t = result["ticker"]
    arrow = "🔺" if t["change_24h"] >= 0 else "🔻"

    text = (
        "┏━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
        "┃  📊 *تحليل البتكوين الاحترافي*  ┃\n"
        "┗━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n"
        f"💰 السعر: *${t['price']:,.2f}*  (Binance)\n"
        f"{arrow} 24س: {t['change_24h']:+.2f}%\n"
        f"⬆️ أعلى 24س: ${t['high_24h']:,.2f}   ⬇️ أدنى 24س: ${t['low_24h']:,.2f}\n"
        f"📈 حجم التداول (24س): ${t['volume_24h']:,.0f}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "*المؤشرات الفنية (Binance 1H):*\n"
        + "\n".join(f"  {sig}" for sig in result["signals"])
        + "\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"*الخلاصة:* {result['verdict']}\n"
        f"(قوة الإشارة الموزونة: {result['score']:+.1f})\n\n"
        "⚠️ _تحليل آلي مبني على مؤشرات تاريخية فقط — ليس توصية استثمارية "
        "أو مالية. اتخاذ القرار مسؤوليتك الشخصية._"
    )
    return text


# ══════════════════════════════════════════════════════════════════
#  رسم الشموع اليابانية الحقيقية
# ══════════════════════════════════════════════════════════════════
def generate_candlestick_chart(df: pd.DataFrame, path: str = CHART_PATH):
    plot_df = df.tail(100)  # آخر 100 شمعة عشان الرسم يكون واضح

    add_plots = [
        mpf.make_addplot(plot_df["ema20"], color="orange", width=1.0),
        mpf.make_addplot(plot_df["ema50"], color="blue", width=1.0),
        mpf.make_addplot(plot_df["bb_upper"], color="gray", width=0.7, linestyle="--"),
        mpf.make_addplot(plot_df["bb_lower"], color="gray", width=0.7, linestyle="--"),
    ]

    mpf.plot(
        plot_df,
        type="candle",
        style="charles",
        addplot=add_plots,
        volume=True,
        title=f"\nBTC/USDT — {TIMEFRAME}",
        ylabel="السعر (USDT)",
        ylabel_lower="الحجم",
        savefig=dict(fname=path, dpi=150, bbox_inches="tight"),
        figsize=(11, 7),
    )
    return path


# ══════════════════════════════════════════════════════════════════
#  لوحة الأزرار
# ══════════════════════════════════════════════════════════════════
def main_menu_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("💰 السعر", callback_data="price"),
            InlineKeyboardButton("📊 تحليل كامل", callback_data="analyze"),
        ],
        [
            InlineKeyboardButton("🕯️ شارت الشموع", callback_data="chart"),
            InlineKeyboardButton("🔔 تنبيهاتي", callback_data="my_alerts"),
        ],
        [
            InlineKeyboardButton("⏱️ الاشتراك الدوري", callback_data="sub_menu"),
            InlineKeyboardButton("❓ المساعدة", callback_data="help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# ══════════════════════════════════════════════════════════════════
#  الأوامر
# ══════════════════════════════════════════════════════════════════
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *أهلاً بك في بوت البتكوين الاحترافي — Ultra Edition*\n\n"
        "بيانات مباشرة من Binance، مؤشرات فنية متقدمة (RSI, MACD, "
        "EMA, Bollinger, Stochastic RSI, ADX)، ورسم شموع يابانية حقيقي.\n\n"
        "اختر من القائمة 👇",
        parse_mode="Markdown",
        reply_markup=main_menu_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *دليل الأوامر*\n\n"
        "/price — السعر الحالي (Binance)\n"
        "/analyze — تحليل فني شامل موزون\n"
        "/chart — صورة شموع يابانية + مؤشرات\n"
        "/alert 65000 above — تنبيه لما السعر يتجاوز 65000$\n"
        "/alert 60000 below — تنبيه لما ينزل تحت 60000$\n"
        "/myalerts — عرض تنبيهاتك\n"
        "/clearalerts — حذف كل تنبيهاتك\n"
        "/subscribe 15 — تحديث تلقائي كل 15 دقيقة\n"
        "/unsubscribe — إيقاف التحديث التلقائي\n"
        "/menu — القائمة التفاعلية\n\n"
        "⚠️ كل التحليلات آلية ولأغراض معلوماتية فقط."
    )
    target = update.message or update.callback_query.message
    await target.reply_text(text, parse_mode="Markdown")


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("القائمة الرئيسية 👇", reply_markup=main_menu_keyboard())


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        t = fetch_ticker_snapshot()
        arrow = "🔺" if t["change_24h"] >= 0 else "🔻"
        text = (
            f"💰 *سعر البتكوين* (Binance): ${t['price']:,.2f}\n"
            f"{arrow} التغيّر (24س): {t['change_24h']:+.2f}%\n"
            f"⬆️ أعلى: ${t['high_24h']:,.2f}   ⬇️ أدنى: ${t['low_24h']:,.2f}"
        )
        target = update.message or update.callback_query.message
        await target.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.exception("price_command failed")
        target = update.message or update.callback_query.message
        await target.reply_text(f"⚠️ خطأ أثناء جلب السعر: {e}")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        result = build_full_analysis()
        text = format_analysis_message(result)
        target = update.message or update.callback_query.message
        await target.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.exception("analyze_command failed")
        target = update.message or update.callback_query.message
        await target.reply_text(f"⚠️ خطأ أثناء التحليل: {e}")


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = update.message or update.callback_query.message
    try:
        df = fetch_ohlcv_df()
        df = enrich_with_indicators(df)
        path = generate_candlestick_chart(df)
        with open(path, "rb") as img:
            await target.reply_photo(
                photo=InputFile(img),
                caption="🕯️ BTC/USDT — 1H (آخر 100 شمعة)\nخط برتقالي = EMA20 | خط أزرق = EMA50 | متقطع = Bollinger Bands",
            )
    except Exception as e:
        logger.exception("chart_command failed")
        await target.reply_text(f"⚠️ خطأ أثناء توليد الشارت: {e}")


async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) != 2 or args[1].lower() not in ("above", "below"):
        await update.message.reply_text(
            "الاستخدام:\n`/alert 65000 above`\n`/alert 60000 below`",
            parse_mode="Markdown",
        )
        return
    try:
        target_price = float(args[0])
    except ValueError:
        await update.message.reply_text("الرجاء إدخال رقم صحيح للسعر.")
        return

    direction = args[1].lower()
    chat_id = update.effective_chat.id
    storage.add_alert(chat_id, target_price, direction)

    emoji = "⬆️" if direction == "above" else "⬇️"
    await update.message.reply_text(
        f"✅ تم ضبط التنبيه: {emoji} عند ${target_price:,.2f}"
    )


async def myalerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    active = [a for a in storage.get_alerts(chat_id) if not a["triggered"]]

    target = update.message or update.callback_query.message
    if not active:
        await target.reply_text("ما عندك تنبيهات مفعّلة حاليًا. استخدم /alert لإضافة واحد.")
        return

    lines = [f"{'⬆️' if a['direction']=='above' else '⬇️'} ${a['price']:,.2f}" for a in active]
    await target.reply_text("🔔 *تنبيهاتك المفعّلة:*\n" + "\n".join(lines), parse_mode="Markdown")


async def clearalerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    storage.clear_alerts(chat_id)
    await update.message.reply_text("🗑️ تم حذف كل تنبيهاتك.")


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    minutes = DEFAULT_AUTO_INTERVAL // 60
    if context.args:
        try:
            minutes = max(5, int(context.args[0]))
        except ValueError:
            pass

    interval_seconds = minutes * 60
    job_name = f"auto_update_{chat_id}"

    for job in context.job_queue.get_jobs_by_name(job_name):
        job.schedule_removal()

    context.job_queue.run_repeating(
        send_auto_update, interval=interval_seconds, first=5, chat_id=chat_id, name=job_name
    )
    storage.add_subscription(chat_id, interval_seconds)

    target = update.message or update.callback_query.message
    await target.reply_text(f"✅ تم تفعيل التحديث التلقائي كل {minutes} دقيقة.")


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    job_name = f"auto_update_{chat_id}"
    jobs = context.job_queue.get_jobs_by_name(job_name)

    target = update.message or update.callback_query.message
    if not jobs:
        await target.reply_text("ما عندك اشتراك مفعّل أصلاً.")
        return
    for job in jobs:
        job.schedule_removal()
    storage.remove_subscription(chat_id)
    await target.reply_text("❌ تم إيقاف التحديث التلقائي.")


# ══════════════════════════════════════════════════════════════════
#  الأزرار التفاعلية
# ══════════════════════════════════════════════════════════════════
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "price":
        await price_command(update, context)
    elif query.data == "analyze":
        await analyze_command(update, context)
    elif query.data == "chart":
        await chart_command(update, context)
    elif query.data == "my_alerts":
        await myalerts_command(update, context)
    elif query.data == "help":
        await help_command(update, context)
    elif query.data == "sub_menu":
        await query.message.reply_text(
            "لتفعيل التحديث الدوري استخدم:\n`/subscribe 15`",
            parse_mode="Markdown",
        )


# ══════════════════════════════════════════════════════════════════
#  المهام الدورية
# ══════════════════════════════════════════════════════════════════
async def send_auto_update(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    try:
        result = build_full_analysis()
        text = format_analysis_message(result)
        await context.bot.send_message(chat_id=job.chat_id, text=text, parse_mode="Markdown")
    except Exception:
        logger.exception("فشل التحديث التلقائي لـ chat_id=%s", job.chat_id)


async def check_alerts_job(context: ContextTypes.DEFAULT_TYPE):
    try:
        current_price = fetch_ticker_snapshot()["price"]
    except Exception:
        logger.exception("check_alerts_job: فشل جلب السعر")
        return

    for chat_id_str, alerts in storage.all_alerts().items():
        chat_id = int(chat_id_str)
        for idx, alert in enumerate(alerts):
            if alert["triggered"]:
                continue
            hit = (
                (alert["direction"] == "above" and current_price >= alert["price"])
                or (alert["direction"] == "below" and current_price <= alert["price"])
            )
            if hit:
                storage.mark_triggered(chat_id, idx)
                emoji = "🚀" if alert["direction"] == "above" else "⚠️"
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"{emoji} *تنبيه سعري!*\n"
                            f"وصل سعر البتكوين إلى ${current_price:,.2f}\n"
                            f"(الهدف: ${alert['price']:,.2f})"
                        ),
                        parse_mode="Markdown",
                    )
                except Exception:
                    logger.exception("فشل إرسال تنبيه لـ chat_id=%s", chat_id)


def restore_subscriptions(app: Application):
    for chat_id_str, interval in storage.all_subscriptions().items():
        chat_id = int(chat_id_str)
        app.job_queue.run_repeating(
            send_auto_update,
            interval=interval,
            first=10,
            chat_id=chat_id,
            name=f"auto_update_{chat_id}",
        )
    logger.info("تم استعادة %d اشتراك دوري.", len(storage.all_subscriptions()))


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("حدث خطأ غير متوقع: %s", context.error, exc_info=context.error)


# ══════════════════════════════════════════════════════════════════
#  نقطة التشغيل
# ══════════════════════════════════════════════════════════════════
def main():
    if not BOT_TOKEN:
        raise SystemExit("لم يتم ضبط متغير البيئة TELEGRAM_BOT_TOKEN. أضفه من تبويب Variables في Railway.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("chart", chart_command))
    app.add_handler(CommandHandler("alert", alert_command))
    app.add_handler(CommandHandler("myalerts", myalerts_command))
    app.add_handler(CommandHandler("clearalerts", clearalerts_command))
    app.add_handler(CommandHandler("subscribe", subscribe_command))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(error_handler)

    app.job_queue.run_repeating(check_alerts_job, interval=ALERT_CHECK_INTERVAL, first=15)
    restore_subscriptions(app)

    logger.info("🚀 البوت Ultra بدأ العمل عند %s", datetime.now())
    app.run_polling()


if __name__ == "__main__":
    main()
