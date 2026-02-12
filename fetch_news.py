#!/usr/bin/env python3
"""
AI Pulse — News Fetcher Backend
Fetches AI news from RSS feeds + X/Twitter via Anthropic API web search.
Summarizes everything with Claude and saves to news.json.

Run via cron 3x daily at 07:00, 13:00, 18:00 CET:
  0 7,13,18 * * * cd /var/www/ai-pulse && /usr/bin/python3 fetch_news.py

Requirements:
  pip install feedparser requests anthropic
"""

import os
import sys
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

# ─── CONFIG ───
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent
NEWS_FILE = OUTPUT_DIR / "news.json"
LOG_FILE = OUTPUT_DIR / "fetch.log"
MAX_ARTICLES_PER_FEED = 10
MAX_DAYS_TO_KEEP = 7
MODEL = "claude-sonnet-4-20250514"

# ─── LOGGING ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ai-pulse")

# ─── RSS FEEDS ───
RSS_FEEDS = [
    {"name": "The Verge", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "tag": "AI"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "tag": "AI"},
    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "tag": "Tech"},
    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/", "tag": "Research"},
    {"name": "VentureBeat", "url": "https://venturebeat.com/category/ai/feed/", "tag": "AI"},
    {"name": "Wired", "url": "https://www.wired.com/feed/tag/ai/latest/rss", "tag": "AI"},
    {"name": "Reuters", "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best", "tag": "Business"},
    {"name": "Hacker News", "url": "https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT+OR+Claude+OR+machine+learning&count=15", "tag": "Community"},
]

AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "llm", "gpt", "claude",
    "openai", "anthropic", "deepmind", "neural", "transformer", "chatbot",
    "generative", "deep learning", "robot", "autonomous", "copilot", "gemini",
    "mistral", "meta ai", "llama", "diffusion", "nvidia", "gpu", "chip",
    "model", "training", "inference", "benchmark", "alignment", "safety",
    "hugging face", "stable diffusion", "midjourney", "sora", "agent",
]

# ─── CET TIME HELPERS ───
CET = timezone(timedelta(hours=1))

def get_cet_now():
    return datetime.now(CET)

def get_slot_label():
    h = get_cet_now().hour
    if h >= 18:
        return "Evening Digest — 18:00 CET"
    elif h >= 13:
        return "Afternoon Update — 13:00 CET"
    else:
        return "Morning Brief — 07:00 CET"

def get_cet_date():
    return get_cet_now().strftime("%Y-%m-%d")


# ─── RSS FETCHING ───
def fetch_rss_feed(feed):
    """Fetch and parse a single RSS feed."""
    log.info(f"  Fetching {feed['name']}...")
    try:
        parsed = feedparser.parse(feed["url"])
        articles = []
        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            title = entry.get("title", "").strip()
            if not title:
                continue

            # Get description, strip HTML
            desc = entry.get("summary", "") or entry.get("description", "")
            import re
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            desc = re.sub(r"&\w+;", " ", desc).strip()[:500]

            link = entry.get("link", "")
            pub_date = entry.get("published", "") or entry.get("updated", "")

            # Filter for AI relevance
            text = (title + " " + desc).lower()
            is_ai = any(kw in text for kw in AI_KEYWORDS)
            if not is_ai and feed["tag"] != "AI":
                continue

            articles.append({
                "title": title,
                "description": desc,
                "link": link,
                "pub_date": pub_date,
                "source": feed["name"],
                "tag": feed["tag"],
            })

        log.info(f"    → {len(articles)} relevant articles")
        return articles
    except Exception as e:
        log.warning(f"    → Failed: {e}")
        return []


def fetch_all_rss():
    """Fetch all RSS feeds."""
    log.info("Fetching RSS feeds...")
    all_articles = []
    for feed in RSS_FEEDS:
        articles = fetch_rss_feed(feed)
        all_articles.extend(articles)
    log.info(f"Total RSS articles: {len(all_articles)}")
    return all_articles


# ─── ANTHROPIC API ───
def call_anthropic(system_prompt, user_message, use_web_search=False):
    """Call the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        log.error("No ANTHROPIC_API_KEY set!")
        return None

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    body = {
        "model": MODEL,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    if use_web_search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract text from content blocks
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return text
    except Exception as e:
        log.error(f"Anthropic API error: {e}")
        return None


def summarize_articles(articles):
    """Use Claude to summarize and tag articles."""
    if not articles:
        return []

    log.info(f"Summarizing {len(articles)} articles...")
    article_texts = "\n".join(
        f'[{i+1}] "{a["title"]}" ({a["source"]}): {a["description"][:200]}'
        for i, a in enumerate(articles[:18])
    )

    system = """You are a concise AI news summarizer. For each article, produce a JSON array of objects with:
- "index" (1-based, matching input)
- "summary" (1-2 clear sentences, max 150 characters)
- "tags" (array of 1-2 short category tags like "OpenAI", "LLM", "Research", "Hardware", "Regulation", "Agents", "Robotics", "Creative AI", "Healthcare", "Market", "Startup", "Google", "Meta", "Microsoft", "Apple", "NVIDIA")
Return ONLY the JSON array. No markdown fences, no explanation."""

    text = call_anthropic(system, f"Summarize these AI news articles:\n{article_texts}")
    if not text:
        return []

    try:
        import re
        # Find JSON array in response
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in summaries: {e}")

    return []


def fetch_x_news():
    """Use Claude + web search to find latest AI news from X/Twitter."""
    log.info("Fetching AI news from X/Twitter via web search...")

    system = """You search X (Twitter) for the latest AI-related posts and announcements. Find real, specific, recent posts. Return ONLY a JSON array (no markdown, no backticks) of 5-8 objects, each with:
- "headline": concise title summarizing the post (string)
- "summary": what was announced or discussed, max 150 chars (string)
- "source": format as "X / @handle" (string)
- "tags": array of 1-2 tags (string array)
- "link": URL to the tweet/post if available, or empty string
Return ONLY the JSON array."""

    today = get_cet_now().strftime("%B %d, %Y")
    user_msg = f"""Search X/Twitter for the most recent AI news and announcements from today ({today}) or the past 24 hours. Look for posts from @AnthropicAI, @OpenAI, @GoogleDeepMind, @xai, @ylecun, @MistralAI, @nvidia, @huggingface, @sama, @demishassabis, @karpathy, and other prominent AI accounts and researchers. What are the most noteworthy AI-related posts and announcements?"""

    text = call_anthropic(system, user_msg, use_web_search=True)
    if not text:
        return []

    try:
        import re
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            items = json.loads(match.group(0))
            log.info(f"  → {len(items)} X posts found")
            return items
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in X news: {e}")

    return []


# ─── BUILD NEWS JSON ───
def make_article_id(headline, source, date):
    """Generate a stable ID for deduplication."""
    raw = f"{headline}:{source}:{date}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_news_items(rss_articles, summaries, x_items):
    """Combine RSS + summaries + X items into final news format."""
    slot_label = get_slot_label()
    today = get_cet_date()
    items = []

    # RSS articles with AI summaries
    for i, article in enumerate(rss_articles[:18]):
        sum_data = next((s for s in summaries if s.get("index") == i + 1), None)
        item = {
            "id": make_article_id(article["title"], article["source"], today),
            "headline": article["title"],
            "summary": sum_data["summary"] if sum_data else article["description"][:150],
            "description": article["description"],
            "tags": sum_data["tags"] if sum_data else [article.get("tag", "AI")],
            "source": article["source"],
            "link": article.get("link", ""),
            "time": slot_label,
            "date": today,
            "readTime": f"{2 + (i % 5)} min read",
            "isFromX": False,
        }
        items.append(item)

    # X/Twitter items
    for i, xi in enumerate(x_items):
        item = {
            "id": make_article_id(xi.get("headline", ""), xi.get("source", "X"), today),
            "headline": xi.get("headline", "AI Update"),
            "summary": xi.get("summary", ""),
            "description": xi.get("summary", ""),
            "tags": xi.get("tags", ["X", "AI"]),
            "source": xi.get("source", "X / AI Community"),
            "link": xi.get("link", ""),
            "time": slot_label,
            "date": today,
            "readTime": f"{1 + (i % 3)} min read",
            "isFromX": True,
        }
        items.append(item)

    return items


def load_existing_news():
    """Load existing news.json."""
    if NEWS_FILE.exists():
        try:
            with open(NEWS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def prune_old_news(news):
    """Remove news older than MAX_DAYS_TO_KEEP days."""
    cutoff = (get_cet_now() - timedelta(days=MAX_DAYS_TO_KEEP)).strftime("%Y-%m-%d")
    return [n for n in news if n.get("date", "") >= cutoff]


def deduplicate(news):
    """Remove duplicate articles by ID."""
    seen = set()
    unique = []
    for item in news:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)
    return unique


def save_news(news):
    """Save news to JSON file."""
    with open(NEWS_FILE, "w") as f:
        json.dump(news, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(news)} articles to {NEWS_FILE}")


# ─── MAIN ───
def main():
    log.info("=" * 60)
    log.info(f"AI Pulse fetch starting — {get_cet_now().strftime('%Y-%m-%d %H:%M CET')}")
    log.info(f"Slot: {get_slot_label()}")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY environment variable not set!")
        log.error("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # 1. Fetch RSS
    rss_articles = fetch_all_rss()

    # 2. Summarize with Claude
    summaries = summarize_articles(rss_articles)

    # 3. Fetch X/Twitter news
    x_items = fetch_x_news()

    # 4. Build new items
    new_items = build_news_items(rss_articles, summaries, x_items)
    log.info(f"New items this cycle: {len(new_items)} ({len(new_items) - len(x_items)} RSS + {len(x_items)} X)")

    # 5. Merge with existing, prune, deduplicate
    existing = load_existing_news()
    combined = new_items + existing
    combined = deduplicate(combined)
    combined = prune_old_news(combined)

    # Sort: newest date first, then by time slot
    slot_order = {
        "Evening Digest — 18:00 CET": 0,
        "Afternoon Update — 13:00 CET": 1,
        "Morning Brief — 07:00 CET": 2,
    }
    combined.sort(key=lambda x: (x["date"], -slot_order.get(x["time"], 9)), reverse=True)

    # 6. Save
    save_news(combined)

    log.info("=" * 60)
    log.info("Fetch complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
