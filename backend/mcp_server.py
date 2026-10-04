import os
import json
import httpx
from jinja2 import Template
from fastmcp import FastMCP
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from backend or root directory
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
load_dotenv(os.path.join(root_dir, ".env"), override=True)
load_dotenv(override=True)

# Initialize FastMCP Server
mcp = FastMCP("GitHub Dev Card Tools")

# Define Pydantic Schema for Gemini Output
class ProfileAnalysis(BaseModel):
    developer_vibe: str
    top_skills: list[str]
    fun_fact: str
    card_theme: str  # hacker, builder, researcher, designer, open-source-hero

@mcp.tool
def scrape_github(username: str) -> dict:
    """
    Scrape public data for a given GitHub username.
    Returns general profile statistics, top repositories, and aggregated language statistics.
    """
    headers = {
        "User-Agent": "GitHub-Dev-Card-Generator",
        "Accept": "application/vnd.github.v3+json"
    }
    # Optional authentication to bypass API limits
    token = os.environ.get("GITHUB_TOKEN")
    if token and token.strip() and token.strip() != "your_github_token_here":
        headers["Authorization"] = f"Bearer {token.strip()}"
        
    user_url = f"https://api.github.com/users/{username}"
    
    with httpx.Client(headers=headers) as client:
        # Fetch profile
        user_res = client.get(user_url)
        if user_res.status_code == 404:
            raise ValueError(f"GitHub user '{username}' not found.")
        elif user_res.status_code != 200:
            raise Exception(f"Failed to fetch user profile: {user_res.status_code} - {user_res.text}")
        
        user_data = user_res.json()
        
        # Fetch repos
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
        repos_res = client.get(repos_url)
        if repos_res.status_code != 200:
            raise Exception(f"Failed to fetch user repositories: {repos_res.status_code}")
            
        repos = repos_res.json()

    # Sort repos by stars descending and take top 6
    sorted_repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)
    top_6 = []
    for r in sorted_repos[:6]:
        top_6.append({
            "name": r.get("name"),
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language") or "Unknown",
            "description": r.get("description") or "No description provided."
        })
        
    # Aggregate languages across all repos
    languages = {}
    for r in repos:
        lang = r.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
            
    # Sort languages by count
    sorted_langs = dict(sorted(languages.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "username": username,
        "name": user_data.get("name") or username,
        "bio": user_data.get("bio") or "",
        "location": user_data.get("location") or "Unknown",
        "public_repos": user_data.get("public_repos", 0),
        "followers": user_data.get("followers", 0),
        "avatar_url": user_data.get("avatar_url", ""),
        "top_repos": top_6,
        "languages": sorted_langs
    }

@mcp.tool
def analyze_profile(github_data: dict) -> dict:
    """
    Analyze GitHub profile data using Gemini 2.5 Flash to extract developer personality traits.
    Returns developer vibe, top 3 skills, a clever fun fact, and a matched card theme.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "your_gemini_api_key_here":
        # Fallback profile analysis if no API key is set for local testing
        return {
            "developer_vibe": "An awesome builder exploring the digital frontier.",
            "top_skills": list(github_data.get("languages", {}).keys())[:3] or ["Python", "TypeScript", "HTML"],
            "fun_fact": "Likes building neat things in the open-source community.",
            "card_theme": "builder"
        }
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Analyze the following GitHub developer profile data:
    {json.dumps(github_data, indent=2)}
    
    Generate a JSON output structured as follows:
    - developer_vibe: A witty or professional 1-sentence summary of their developer style.
    - top_skills: Exactly 3 primary technologies, languages, or skills.
    - fun_fact: A clever, lighthearted deduction based on their projects/languages/names.
    - card_theme: Pick exactly one theme from this list: "hacker", "builder", "researcher", "designer", "open-source-hero".
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProfileAnalysis,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        # Graceful fallback in case of API failure or format issues
        return {
            "developer_vibe": f"A prolific developer based in {github_data.get('location', 'the cloud')}.",
            "top_skills": list(github_data.get("languages", {}).keys())[:3] or ["Python", "JavaScript", "GitHub"],
            "fun_fact": "Has public repositories spanning multiple interesting topics.",
            "card_theme": "open-source-hero"
        }

@mcp.tool
def generate_card_html(username: str, github_data: dict, analysis: dict) -> str:
    """
    Generate a complete self-contained HTML page representing the developer card.
    Applies custom CSS variables based on the selected theme.
    """
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ name }}'s Dev Card</title>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
  <style>
    /* Theme definitions */
    :root {
      --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      --card-bg: rgba(30, 41, 59, 0.7);
      --accent-color: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.4);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-color: rgba(255, 255, 255, 0.08);
      --font-family: 'Outfit', sans-serif;
      --badge-bg: rgba(56, 189, 248, 0.15);
      --badge-text: #38bdf8;
    }
    
    .theme-hacker {
      --bg-gradient: linear-gradient(135deg, #020617 0%, #090d16 100%);
      --card-bg: rgba(9, 13, 22, 0.85);
      --accent-color: #22c55e;
      --accent-glow: rgba(34, 197, 94, 0.4);
      --text-main: #f0fdf4;
      --text-muted: #86efac;
      --border-color: rgba(34, 197, 94, 0.25);
      --font-family: 'Fira Code', monospace;
      --badge-bg: rgba(34, 197, 94, 0.15);
      --badge-text: #22c55e;
    }
    
    .theme-builder {
      --bg-gradient: linear-gradient(135deg, #0b1329 0%, #1c2541 100%);
      --card-bg: rgba(28, 37, 65, 0.8);
      --accent-color: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.4);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --border-color: rgba(59, 130, 246, 0.2);
      --font-family: 'Outfit', sans-serif;
      --badge-bg: rgba(59, 130, 246, 0.15);
      --badge-text: #60a5fa;
    }

    .theme-researcher {
      --bg-gradient: linear-gradient(135deg, #180030 0%, #0c001c 100%);
      --card-bg: rgba(24, 0, 48, 0.7);
      --accent-color: #a855f7;
      --accent-glow: rgba(168, 85, 247, 0.4);
      --text-main: #faf5ff;
      --text-muted: #d8b4fe;
      --border-color: rgba(168, 85, 247, 0.2);
      --font-family: 'Outfit', sans-serif;
      --badge-bg: rgba(168, 85, 247, 0.15);
      --badge-text: #c084fc;
    }

    .theme-designer {
      --bg-gradient: linear-gradient(135deg, #1f0010 0%, #000000 100%);
      --card-bg: rgba(31, 0, 16, 0.75);
      --accent-color: #ec4899;
      --accent-glow: rgba(236, 72, 153, 0.4);
      --text-main: #fdf2f8;
      --text-muted: #f472b6;
      --border-color: rgba(236, 72, 153, 0.2);
      --font-family: 'Outfit', sans-serif;
      --badge-bg: rgba(236, 72, 153, 0.15);
      --badge-text: #f472b6;
    }

    .theme-open-source-hero {
      --bg-gradient: linear-gradient(135deg, #1c0d02 0%, #0f0500 100%);
      --card-bg: rgba(28, 13, 2, 0.75);
      --accent-color: #f97316;
      --accent-glow: rgba(249, 115, 22, 0.4);
      --text-main: #fff7ed;
      --text-muted: #ffb076;
      --border-color: rgba(249, 115, 22, 0.2);
      --font-family: 'Outfit', sans-serif;
      --badge-bg: rgba(249, 115, 22, 0.15);
      --badge-text: #ff9d5c;
    }

    /* Core Styles */
    body {
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      background: var(--bg-gradient);
      font-family: var(--font-family);
      color: var(--text-main);
      overflow-y: auto;
    }

    /* Premium card wrapper with animated glow */
    .card-container {
      position: relative;
      width: 420px;
      margin: 2rem auto;
      padding: 2.5rem 2rem;
      border-radius: 24px;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 40px var(--accent-glow);
      transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .card-container:hover {
      transform: translateY(-8px);
      box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5), 0 0 60px var(--accent-glow);
      border-color: var(--accent-color);
    }

    /* Header Profile Info */
    .profile-header {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }

    .avatar {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      border: 3px solid var(--accent-color);
      box-shadow: 0 0 15px var(--accent-glow);
      object-fit: cover;
    }

    .info {
      display: flex;
      flex-direction: column;
    }

    .name {
      font-size: 1.5rem;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.5px;
    }

    .username {
      font-size: 0.95rem;
      color: var(--accent-color);
      text-decoration: none;
      margin-top: 0.2rem;
      font-weight: 600;
    }

    /* Vibe and description */
    .vibe {
      font-size: 1rem;
      line-height: 1.5;
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.03);
      padding: 0.8rem 1.2rem;
      border-radius: 12px;
      border-left: 4px solid var(--accent-color);
      margin: 1.2rem 0;
      font-style: italic;
    }

    /* Stats Row */
    .stats-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin: 1.5rem 0;
    }

    .stat-box {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border-color);
      padding: 0.8rem;
      border-radius: 12px;
      text-align: center;
    }

    .stat-val {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--accent-color);
      display: block;
    }

    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 0.2rem;
    }

    /* Skills Badges */
    .skills-container {
      margin-bottom: 1.5rem;
    }

    .section-title {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: var(--text-muted);
      margin-bottom: 0.8rem;
      font-weight: 600;
    }

    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .badge {
      background: var(--badge-bg);
      color: var(--badge-text);
      padding: 0.4rem 0.8rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Top Projects */
    .projects-container {
      margin-top: 1.5rem;
    }

    .project-card {
      background: rgba(255, 255, 255, 0.015);
      border: 1px solid var(--border-color);
      padding: 0.9rem;
      border-radius: 12px;
      margin-bottom: 0.8rem;
      transition: all 0.3s ease;
    }

    .project-card:hover {
      background: rgba(255, 255, 255, 0.04);
      border-color: var(--accent-color);
      transform: scale(1.01);
    }

    .project-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
    }

    .project-name {
      font-size: 0.95rem;
      font-weight: 600;
      margin: 0;
      color: var(--text-main);
    }

    .project-stars {
      display: flex;
      align-items: center;
      gap: 0.25rem;
      font-size: 0.8rem;
      color: #fbbf24;
      font-weight: 600;
    }

    .project-desc {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin: 0 0 0.5rem 0;
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .project-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
    }

    .project-lang {
      color: var(--accent-color);
      font-weight: 600;
    }
    
    .fun-fact {
      margin-top: 1.5rem;
      padding-top: 1rem;
      border-top: 1px dashed var(--border-color);
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.5;
    }

    .fun-fact strong {
      color: var(--accent-color);
    }
  </style>
</head>
<body class="theme-{{ card_theme }}">

  <div class="card-container">
    <div class="profile-header">
      <img src="{{ avatar_url }}" alt="{{ name }}" class="avatar" onerror="this.src='https://github.com/identicons/{{ username }}.png'">
      <div class="info">
        <h1 class="name">{{ name }}</h1>
        <a href="https://github.com/{{ username }}" target="_blank" class="username">@{{ username }}</a>
      </div>
    </div>

    <div class="vibe">
      &ldquo;{{ developer_vibe }}&rdquo;
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <span class="stat-val">{{ public_repos }}</span>
        <span class="stat-label">Repositories</span>
      </div>
      <div class="stat-box">
        <span class="stat-val">{{ followers }}</span>
        <span class="stat-label">Followers</span>
      </div>
    </div>

    <div class="skills-container">
      <div class="section-title">Top Skills</div>
      <div class="badges">
        {% for skill in top_skills %}
        <span class="badge">{{ skill }}</span>
        {% endfor %}
      </div>
    </div>

    <div class="projects-container">
      <div class="section-title">Featured Projects</div>
      {% for repo in top_repos %}
      <div class="project-card">
        <div class="project-header">
          <h3 class="project-name">{{ repo.name }}</h3>
          <span class="project-stars">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/></svg>
            {{ repo.stars }}
          </span>
        </div>
        <p class="project-desc">{{ repo.description }}</p>
        <div class="project-meta">
          <span class="project-lang">{{ repo.language }}</span>
        </div>
      </div>
      {% endfor %}
    </div>
    
    <div class="fun-fact">
      <strong>Fun Fact:</strong> {{ fun_fact }}
    </div>
  </div>

</body>
</html>"""

    # Take only the top 3 repos for the card as specified
    top_3_repos = github_data.get("top_repos", [])[:3]
    
    # Render with Jinja2
    template = Template(html_template)
    rendered_html = template.render(
        name=github_data.get("name"),
        username=username,
        avatar_url=github_data.get("avatar_url"),
        developer_vibe=analysis.get("developer_vibe"),
        public_repos=github_data.get("public_repos"),
        followers=github_data.get("followers"),
        top_skills=analysis.get("top_skills", []),
        top_repos=top_3_repos,
        fun_fact=analysis.get("fun_fact"),
        card_theme=analysis.get("card_theme", "builder")
    )
    
    return rendered_html

@mcp.tool
def save_card(username: str, html: str) -> str:
    """
    Saves the rendered card HTML string to a static directory.
    Returns the relative path to access the card.
    """
    # Define directory path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static", "cards")
    os.makedirs(static_dir, exist_ok=True)
    
    file_name = f"{username.lower()}.html"
    file_path = os.path.join(static_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return f"/static/cards/{file_name}"

if __name__ == "__main__":
    mcp.run()
