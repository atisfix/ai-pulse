#!/usr/bin/env python3
"""
AI Pulse — News Fetcher Backend
Fetches AI news from RSS feeds + X/Twitter via Anthropic API web search.
Summarizes everything with Claude and saves to news.json.

Schedule (GitHub Actions cron, all in UTC):
  06:00 UTC = 07:00 CET → Morning Brief
  12:00 UTC = 13:00 CET → Afternoon Update
  17:00 UTC = 18:00 CET → Evening Digest

Requirements:
  pip install feedparser requests anthropic
"""

import os
import sys
import json
import re
import html as html_module
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
SEEN_FILE = OUTPUT_DIR / "seen_articles.json"
LOG_FILE = OUTPUT_DIR / "fetch.log"
MAX_ARTICLES_PER_FEED = 10
MAX_DAYS_TO_KEEP = 30
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
# Tier 1: Major tech news (AI-specific feeds)
RSS_FEEDS = [
    {"name": "The Verge", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "tag": "AI"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "tag": "AI"},
    {"name": "VentureBeat", "url": "https://venturebeat.com/category/ai/feed/", "tag": "AI"},
    {"name": "Wired AI", "url": "https://www.wired.com/feed/tag/ai/latest/rss", "tag": "AI"},
    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "tag": "Tech"},
    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/", "tag": "Research"},
    # Tier 2: Google News AI topic (aggregates from many sources)
    {"name": "Google News AI", "url": "https://news.google.com/rss/topics/CAAqIQgKIhtDQkFTRGdvSUwyMHZNRGRqTVhZU0FtVnVLQUFQAQ?hl=en-US&gl=US&ceid=US:en", "tag": "AI"},
    # Tier 3: AI-specific outlets
    {"name": "AI News", "url": "https://www.artificialintelligence-news.com/feed/rss/", "tag": "AI"},
    {"name": "MarkTechPost", "url": "https://www.marktechpost.com/feed/", "tag": "Research"},
    {"name": "The Guardian AI", "url": "https://www.theguardian.com/technology/artificialintelligenceai/rss", "tag": "AI"},
    {"name": "Futurism AI", "url": "https://futurism.com/categories/ai-artificial-intelligence/feed", "tag": "AI"},
    {"name": "SiliconANGLE", "url": "https://siliconangle.com/category/big-data/feed", "tag": "Tech"},
    # Tier 4: Company & research blogs
    {"name": "OpenAI Blog", "url": "https://openai.com/blog/rss.xml", "tag": "OpenAI"},
    {"name": "Google AI Blog", "url": "https://research.google/blog/rss/", "tag": "Google"},
    {"name": "Anthropic Blog", "url": "https://www.anthropic.com/rss.xml", "tag": "Anthropic"},
    {"name": "DeepMind Blog", "url": "https://deepmind.google/blog/rss.xml", "tag": "DeepMind"},
    {"name": "Meta AI Blog", "url": "https://ai.meta.com/blog/rss/", "tag": "Meta"},
    {"name": "Microsoft AI Blog", "url": "https://blogs.microsoft.com/ai/feed/", "tag": "Microsoft"},
    {"name": "NVIDIA Blog", "url": "https://blogs.nvidia.com/feed/", "tag": "NVIDIA"},
    {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml", "tag": "Open Source"},
    {"name": "MIT News AI", "url": "https://news.mit.edu/rss/topic/artificial-intelligence2", "tag": "Research"},
    # Tier 5: Community & aggregators
    {"name": "Hacker News AI", "url": "https://hnrss.org/newest?q=AI+OR+LLM+OR+GPT+OR+Claude+OR+machine+learning&count=15", "tag": "Community"},
    {"name": "Reddit r/AI", "url": "https://www.reddit.com/r/artificial/.rss", "tag": "Community"},
    {"name": "Science Daily AI", "url": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", "tag": "Science"},
    # Tier 6: Image Gen & Video Gen specific
    {"name": "Reddit r/StableDiffusion", "url": "https://www.reddit.com/r/StableDiffusion/.rss", "tag": "Image Gen"},
    {"name": "Reddit r/midjourney", "url": "https://www.reddit.com/r/midjourney/.rss", "tag": "Image Gen"},
    {"name": "Reddit r/aivideo", "url": "https://www.reddit.com/r/aivideo/.rss", "tag": "Video Gen"},
    {"name": "Google News AI Video", "url": "https://news.google.com/rss/search?q=AI+video+generation+OR+Sora+OR+Runway+OR+Kling+OR+Pika&hl=en-US&gl=US&ceid=US:en", "tag": "Video Gen"},
    {"name": "Google News AI Image", "url": "https://news.google.com/rss/search?q=AI+image+generation+OR+Midjourney+OR+DALL-E+OR+Stable+Diffusion+OR+Flux&hl=en-US&gl=US&ceid=US:en", "tag": "Image Gen"},
]

AI_KEYWORDS = [
    "ai ", " ai", "a.i.", "artificial intelligence", "machine learning", "llm",
    "gpt", "gpt-4", "gpt-5", "claude", "chatgpt",
    "openai", "anthropic", "deepmind", "google ai", "meta ai",
    "neural network", "transformer", "chatbot", "large language model",
    "generative ai", "gen ai", "deep learning", "copilot",
    "gemini", "mistral", "llama", "diffusion model",
    "nvidia ai", "hugging face", "stable diffusion", "midjourney",
    "sora", "dall-e", "dall-e 3", "flux", "imagen",
    "agi", "superintelligence", "foundation model", "fine-tuning", "fine tuning",
    "rlhf", "prompt engineering", "context window", "multimodal",
    "text-to-image", "text-to-video", "text to image", "text to video",
    "image generation", "video generation", "ai image", "ai video", "ai art",
    "speech recognition", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "synthetic data",
    "ai regulation", "ai safety", "ai ethics", "ai startup", "ai chip",
    "ai agent", "ai model", "ai training", "ai inference", "ai benchmark",
    "pytorch", "tensorflow", "ai alignment",
    "groq", "cerebras", "cohere", "perplexity", "cursor ai", "windsurf",
    "runway", "kling", "pika", "controlnet", "lora", "comfyui",
]

# ─── CET TIME HELPERS ───
CET = timezone(timedelta(hours=1))

def get_cet_now():
    return datetime.now(CET)

def get_slot_label():
    """Determine slot based on the cron schedule that triggered us.
    GitHub Actions cron runs at 06, 12, 17 UTC = 07, 13, 18 CET.
    We use UTC hour to determine which slot we are in, since GitHub runs in UTC."""
    utc_hour = datetime.now(timezone.utc).hour
    # 17:00 UTC (18:00 CET) window: 16-20 UTC
    if 16 <= utc_hour <= 20:
        return "Evening Digest \u2014 18:00 CET"
    # 12:00 UTC (13:00 CET) window: 11-15 UTC
    elif 11 <= utc_hour <= 15:
        return "Afternoon Update \u2014 13:00 CET"
    # 06:00 UTC (07:00 CET) window: everything else
    else:
        return "Morning Brief \u2014 07:00 CET"

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

            desc = entry.get("summary", "") or entry.get("description", "")
            desc = html_module.unescape(desc)
            desc = re.sub(r"<[^>]+>", "", desc).strip()
            desc = re.sub(r"\[\s*\.\.\.\s*\]|\[\s*\u2026\s*\]", "...", desc)
            desc = re.sub(r"\s+", " ", desc).strip()[:500]

            title = html_module.unescape(title)
            title = re.sub(r"\[\s*\.\.\.\s*\]|\[\s*\u2026\s*\]", "...", title)

            link = entry.get("link", "")
            pub_date = entry.get("published", "") or entry.get("updated", "")

            text = (title + " " + desc).lower()
            # Pad text with spaces for word-boundary matching
            text_padded = " " + text + " "
            is_ai = any(kw in text_padded for kw in AI_KEYWORDS)
            if not is_ai:
                continue

            articles.append({
                "title": title,
                "description": desc,
                "link": link,
                "pub_date": pub_date,
                "source": feed["name"],
                "tag": feed["tag"],
            })

        log.info(f"    \u2192 {len(articles)} relevant articles")
        return articles
    except Exception as e:
        log.warning(f"    \u2192 Failed: {e}")
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
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
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
        for i, a in enumerate(articles[:25])
    )

    system = """You are a concise AI news summarizer. For each article, produce a JSON array of objects with:
- "index" (1-based, matching input)
- "summary" (1-2 clear sentences, max 150 characters)
- "tags" (array of 1-2 short category tags from this list: "OpenAI", "LLM", "Research", "Hardware", "Regulation", "Agents", "Robotics", "Video Gen", "Image Gen", "Creative AI", "Healthcare", "Market", "Startup", "Google", "Meta", "Microsoft", "Apple", "NVIDIA", "Anthropic", "Open Source")
- "relevant" (boolean: true if the article is specifically about AI/ML, false if it is general tech, business, or unrelated)
Use "Video Gen" for anything about AI video generation (Sora, Runway, Kling, Pika, text-to-video, video synthesis).
Use "Image Gen" for anything about AI image generation (Midjourney, DALL-E, Stable Diffusion, Flux, text-to-image, image synthesis).
IMPORTANT: Set "relevant" to false for articles that are about general tech, gadgets, politics, business, or other non-AI topics. Only set it to true for articles that are specifically about AI, machine learning, or related technology.
Return ONLY the JSON array. No markdown fences, no explanation."""

    text = call_anthropic(system, f"Summarize these AI news articles:\n{article_texts}")
    if not text:
        return []

    try:
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
- "link": direct URL to the tweet/post. Use the format https://x.com/handle/status/ID if you can find it, or empty string if not available
- "posted_at": when the post was made, e.g. "2 hours ago", "today at 14:30", "Feb 12, 2026" — be as specific as possible (string)
Return ONLY the JSON array."""

    today = get_cet_now().strftime("%B %d, %Y")
    cet_time = get_cet_now().strftime("%H:%M CET")
    user_msg = f"""Search X/Twitter for the most recent AI news and announcements from today ({today}, current time {cet_time}) or the past 24 hours. Look for posts from @AnthropicAI, @OpenAI, @GoogleDeepMind, @xai, @ylecun, @MistralAI, @nvidia, @huggingface, @sama, @demishassabis, @karpathy, and other prominent AI accounts and researchers. What are the most noteworthy AI-related posts and announcements? Include direct links to the posts and when they were posted."""

    text = call_anthropic(system, user_msg, use_web_search=True)
    if not text:
        return []

    try:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            items = json.loads(match.group(0))
            log.info(f"  \u2192 {len(items)} X posts found")
            return items
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error in X news: {e}")

    return []


# ─── BUILD NEWS JSON ───
def make_article_id(headline, source, date, slot):
    """Generate a stable ID for deduplication including slot."""
    raw = f"{headline}:{source}:{date}:{slot}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_news_items(rss_articles, summaries, x_items):
    """Combine RSS + summaries + X items into final news format."""
    slot_label = get_slot_label()
    today = get_cet_date()
    items = []

    for i, article in enumerate(rss_articles[:25]):
        sum_data = next((s for s in summaries if s.get("index") == i + 1), None)
        # Skip articles Claude flagged as not AI-relevant
        if sum_data and sum_data.get("relevant") is False:
            continue
        # Parse pub_date to get a proper timestamp
        pub_date_str = article.get("pub_date", "")
        published_at = ""
        if pub_date_str:
            try:
                from email.utils import parsedate_to_datetime
                pd = parsedate_to_datetime(pub_date_str)
                published_at = pd.strftime("%d %b %Y, %H:%M")
            except Exception:
                try:
                    # Try ISO format
                    pd = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    published_at = pd.strftime("%d %b %Y, %H:%M")
                except Exception:
                    published_at = ""

        item = {
            "id": make_article_id(article["title"], article["source"], today, slot_label),
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
            "publishedAt": published_at,
        }
        items.append(item)

    for i, xi in enumerate(x_items):
        posted_at = xi.get("posted_at", "")
        source_name = xi.get("source", "X / AI Community")
        summary_with_time = xi.get("summary", "")
        if posted_at:
            summary_with_time = f"[{posted_at}] {summary_with_time}"

        item = {
            "id": make_article_id(xi.get("headline", ""), source_name, today, slot_label),
            "headline": xi.get("headline", "AI Update"),
            "summary": summary_with_time,
            "description": xi.get("summary", ""),
            "tags": xi.get("tags", ["X", "AI"]),
            "source": source_name,
            "link": xi.get("link", ""),
            "time": slot_label,
            "date": today,
            "readTime": f"{1 + (i % 3)} min read",
            "isFromX": True,
            "postedAt": posted_at,
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


# ─── SEEN ARTICLES TRACKING ───
def make_fingerprint(title, source):
    """Create a fingerprint from headline + source for dedup across fetches."""
    raw = re.sub(r"[^a-z0-9 ]", "", (title or "").lower().strip())
    raw += ":" + (source or "").lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def make_content_hash(description):
    """Hash the description to detect if content has been updated."""
    raw = re.sub(r"\s+", " ", (description or "").lower().strip())
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def load_seen():
    """Load seen articles registry.
    Format: { fingerprint: { "first_seen": date, "content_hash": hash } }
    """
    if SEEN_FILE.exists():
        try:
            with open(SEEN_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_seen(seen):
    """Save seen articles registry."""
    with open(SEEN_FILE, "w") as f:
        json.dump(seen, f, indent=2, ensure_ascii=False)

def prune_seen(seen, days=60):
    """Remove entries older than N days to keep file small."""
    cutoff = (get_cet_now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return {k: v for k, v in seen.items() if v.get("first_seen", "") >= cutoff}

def filter_new_articles(rss_articles, seen):
    """Filter out articles that have been seen before (unless updated).
    Returns (new_articles, updated_seen_dict).
    """
    new = []
    today = get_cet_date()

    for article in rss_articles:
        fp = make_fingerprint(article["title"], article["source"])
        content_hash = make_content_hash(article["description"])

        if fp in seen:
            old_hash = seen[fp].get("content_hash", "")
            if content_hash == old_hash:
                # Same article, same content — skip it
                continue
            else:
                # Same article but content changed — allow it as an update
                log.info(f"    Updated article: {article['title'][:60]}...")
                seen[fp]["content_hash"] = content_hash

        else:
            # Brand new article — register it
            seen[fp] = {
                "first_seen": today,
                "content_hash": content_hash,
            }

        new.append(article)

    return new, seen


# ─── MAIN ───
def main():
    slot_label = get_slot_label()
    today = get_cet_date()

    log.info("=" * 60)
    log.info(f"AI Pulse fetch starting")
    log.info(f"  CET time: {get_cet_now().strftime('%Y-%m-%d %H:%M CET')}")
    log.info(f"  UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    log.info(f"  Slot: {slot_label}")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY environment variable not set!")
        sys.exit(1)

    # 1. Fetch RSS
    rss_articles = fetch_all_rss()

    # 2. Filter out already-seen articles (unless updated)
    seen = load_seen()
    rss_articles, seen = filter_new_articles(rss_articles, seen)
    log.info(f"New/updated RSS articles after dedup: {len(rss_articles)}")

    # 3. Summarize with Claude (only new articles — saves API cost too!)
    summaries = summarize_articles(rss_articles)

    # 4. Fetch X/Twitter news
    x_items = fetch_x_news()

    # 5. Build new items
    new_items = build_news_items(rss_articles, summaries, x_items)
    log.info(f"New items this cycle: {len(new_items)} ({len(new_items) - len(x_items)} RSS + {len(x_items)} X)")

    # 6. Load existing news
    existing = load_existing_news()

    # 7. Remove old entries from the SAME slot today (so they get replaced with fresh ones)
    existing = [
        n for n in existing
        if not (n.get("date") == today and n.get("time") == slot_label)
    ]

    # 8. Merge, deduplicate, prune
    combined = new_items + existing
    combined = deduplicate(combined)
    combined = prune_old_news(combined)

    # 9. Sort: newest date first, then evening > afternoon > morning
    slot_order = {
        "Evening Digest \u2014 18:00 CET": 0,
        "Afternoon Update \u2014 13:00 CET": 1,
        "Morning Brief \u2014 07:00 CET": 2,
    }
    combined.sort(key=lambda x: (x["date"], -slot_order.get(x["time"], 9)), reverse=True)

    # 10. Save news + seen registry
    save_news(combined)
    seen = prune_seen(seen, days=60)
    save_seen(seen)
    log.info(f"Seen articles registry: {len(seen)} entries")

    log.info("=" * 60)
    log.info("Fetch complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
