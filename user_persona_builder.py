import praw
import openai
from dotenv import load_dotenv
import os
import re
import json
from tqdm import tqdm

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

def extract_username(url: str) -> str:
    match = re.search(r'reddit.com/user/([^/]+)', url)
    return match.group(1) if match else None

def fetch_user_data(username: str, limit: int = 100):
    redditor = reddit.redditor(username)
    posts, comments = [], []
    
    try:
        for submission in tqdm(redditor.submissions.new(limit=limit), desc="Fetching Posts"):
            posts.append({
                "title": submission.title,
                "body": submission.selftext,
                "url": submission.permalink
            })
        for comment in tqdm(redditor.comments.new(limit=limit), desc="Fetching Comments"):
            comments.append({
                "body": comment.body,
                "url": comment.permalink
            })
    except Exception as e:
        print(f"Error fetching data: {e}")
    
    return posts, comments

def build_prompt(posts, comments):
    combined_text = "### POSTS ###\n"
    for p in posts:
        combined_text += f"- {p['title']}\n{p['body']}\n(Source: https://reddit.com{p['url']})\n\n"
    
    combined_text += "\n### COMMENTS ###\n"
    for c in comments:
        combined_text += f"- {c['body']}\n(Source: https://reddit.com{c['url']})\n\n"

    return (
        f"You are a social psychologist analyzing a Reddit user. Based on the text below, "
        f"generate a detailed user persona. Include:\n"
        f"- Name (guessed or pseudonym), Age range, Gender (if inferable)\n"
        f"- Interests and Hobbies\n"
        f"- Political/Religious views (if any)\n"
        f"- Personality traits\n"
        f"- Communication style\n"
        f"- Occupation or Educational background (if inferred)\n"
        f"- Citations from posts/comments for each point\n\n"
        f"{combined_text}\n"
        f"---\nGenerate the persona as structured text with headings."
    )

def call_openai(prompt):
    print("🧠 Generating persona with LLM...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in social profiling and behavior analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000
    )
    return response['choices'][0]['message']['content']

def save_output(username, persona_text):
    filename = f"user_persona_{username}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"\n✅ Persona saved to: {filename}")

def main():
    url = input("Enter Reddit user profile URL: ").strip()
    username = extract_username(url)

    if not username:
        print("❌ Invalid Reddit URL.")
        return

    print(f"\n🔍 Analyzing user: {username}")
    posts, comments = fetch_user_data(username)

    if not posts and not comments:
        print("❌ No content found for this user.")
        return

    prompt = build_prompt(posts, comments)
    persona_text = call_openai(prompt)
    save_output(username, persona_text)

if __name__ == "__main__":
    main()
