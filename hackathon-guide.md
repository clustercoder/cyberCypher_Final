# Claude Code Hackathon Guide: From Zero to Winning

A complete setup and strategy guide for using Claude Code Pro at an Agentic AI Hackathon (with RAG, LangGraph, multi-agent systems, and frontend).

---

## Part 1: Installation & First-Time Setup

### 1.1 Install Claude Code

The recommended method is the **native installer** (no Node.js required):

```bash
# macOS / Linux
curl -fsSL https://claude.ai/install.sh | sh

# Windows (PowerShell)
irm https://claude.ai/install.ps1 | iex

# Verify
claude --version
```

> **Alternative (Homebrew on macOS):**
> ```bash
> brew install claude-code
> ```
> Note: Homebrew doesn't auto-update — run `brew upgrade claude-code` manually.

### 1.2 Authenticate

Since you have Claude Code Pro, just run:

```bash
claude
```

It will open your browser for OAuth. Log in with your Pro account. No API key needed — your subscription covers it.

> **Important:** Make sure you do NOT have an `ANTHROPIC_API_KEY` environment variable set, or Claude Code will use that (and bill you per-token via API) instead of your Pro subscription.

### 1.3 Install Node.js (You'll Need It for MCPs)

Even though Claude Code itself no longer requires Node.js, many MCP servers run via `npx`, so install it:

```bash
# Check if you already have it
node --version  # Need 18+

# If not, install from https://nodejs.org (LTS version)
```

### 1.4 Run Diagnostics

```bash
claude doctor
```

This checks for missing dependencies, PATH issues, and authentication status.

---

## Part 2: Understanding Your Plan & Token Economics

### 2.1 What Pro Gives You

- Access to **Sonnet 4.6** (default) and **Opus 4.6**
- Claude Code terminal access
- Shared usage limits between claude.ai and Claude Code
- **5-hour rolling window** — your usage resets 5 hours after your first prompt in a window
- Roughly **10–40 prompts per window** depending on complexity

### 2.2 Hackathon Reality Check

Pro is sufficient for a hackathon, but you'll need to be **disciplined with tokens**. Claude Code calls are much heavier than chat — they include file contexts, system prompts, and multi-step reasoning. Here's how to not run dry mid-hackathon:

**Cost-saving settings** — add to `~/.claude/settings.json`:

```json
{
  "model": "sonnet",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "50"
  }
}
```

This does three things:
1. Defaults to **Sonnet** (cheaper, handles 80%+ of coding tasks well)
2. Reduces hidden thinking tokens from ~32k to 10k per request
3. Compacts context earlier (at 50% instead of 95%), keeping quality up in long sessions

**Model switching strategy:**

| Task | Model | Why |
|------|-------|-----|
| Most coding, file edits, tests | `sonnet` | Fast, cheap, good enough |
| Architecture decisions, complex debugging | `opus` | Deep reasoning needed |
| Quick lookups, simple questions | `haiku` | Cheapest option |

Switch models in-session with `/model opus` or `/model sonnet`.

### 2.3 Token-Saving Habits

- **Use `/clear` between unrelated tasks** — it's free and resets context instantly
- **Use `/compact` at logical breakpoints** (after research, before implementation)
- **Run `/cost`** periodically to monitor your spend
- **Batch your requests** — one long detailed prompt beats 5 back-and-forth refinements
- **Keep files modular** — Claude burns tokens re-reading long files. Aim for hundreds of lines per file, not thousands
- **Don't dump entire codebases** — use `@filename` to reference specific files

---

## Part 3: Pre-Hackathon Configuration

Do all of this **before** the hackathon starts. This is your infrastructure.

### 3.1 Create Your Project & CLAUDE.md

```bash
mkdir hackathon-project && cd hackathon-project
git init
```

Create a `CLAUDE.md` in the project root — this is Claude Code's "memory" of your project:

```markdown
# Hackathon Project: [Your Project Name]

## Tech Stack
- Frontend: Next.js 15 (App Router) + TypeScript + Tailwind CSS
- Backend: Python 3.12 + FastAPI
- AI/Agents: LangGraph + LangChain
- Vector DB: Pinecone / ChromaDB (for RAG)
- Database: Supabase (PostgreSQL)
- Deployment: Vercel (frontend) + Railway (backend)

## Project Structure
- `/frontend` — Next.js app
- `/backend` — FastAPI + LangGraph agents
- `/shared` — Shared types/schemas

## Commands
- Frontend dev: `cd frontend && npm run dev`
- Backend dev: `cd backend && uvicorn main:app --reload`
- Tests: `cd backend && pytest`

## Rules
- Use TypeScript strict mode in frontend
- Use Pydantic v2 models for all API schemas
- All agents must have clear input/output schemas
- Write tests for all agent nodes
- No console.logs in production code
- Commit messages: conventional commits format
```

Keep this **concise** — aim for 50–100 lines. Every line should prevent Claude from making a mistake.

### 3.2 Install the Everything Claude Code Plugin

This gives you 16 agents, 65 skills, and 40 commands out of the box:

```bash
# Open Claude Code, then run:
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

Then install the rules manually (plugins can't distribute rules):

```bash
git clone https://github.com/affaan-m/everything-claude-code.git /tmp/ecc
cd /tmp/ecc

# Install common + language-specific rules
mkdir -p ~/.claude/rules
cp -r rules/common/* ~/.claude/rules/
cp -r rules/typescript/* ~/.claude/rules/
cp -r rules/python/* ~/.claude/rules/
```

### 3.3 Set Up Key MCP Servers

MCPs connect Claude to external services. **Keep under 10 enabled** — each one eats context window.

For your hackathon stack, you'll want:

```bash
# GitHub (for PRs, issues)
claude mcp add github -- npx -y @modelcontextprotocol/server-github

# Web search (for looking up docs mid-session)
claude mcp add web-search -- npx -y @anthropic-ai/mcp-server-web-search

# Supabase (if using it as your DB)
claude mcp add supabase -- npx -y @supabase/mcp-server-supabase@latest --project-ref=YOUR_REF
```

Check what's enabled:

```bash
claude mcp list
```

**Critical rule:** Disable MCPs you're not actively using. Navigate to `/plugins` in Claude Code to toggle them.

### 3.4 Set Up Your Editor

**Recommended: VS Code or Cursor** (since you have Copilot Pro too)

Split your screen:
- **Left:** Editor with your code
- **Right:** Terminal running Claude Code

Enable auto-save in your editor so Claude's file reads are always current.

**How Claude Code + Copilot Pro work together:**
- Use **Copilot** for inline completions as you type (it's great for boilerplate)
- Use **Claude Code** for multi-file changes, architecture, debugging, and agent orchestration
- They don't conflict — Copilot works in-editor, Claude Code works in terminal

### 3.5 Create Subagents for Your Hackathon

Create `~/.claude/agents/` directory with agents tailored to your project:

**`planner.md`** — for breaking down features:
```markdown
---
name: planner
description: Breaks down features into implementation steps
tools: ["Read", "Grep", "Glob"]
model: sonnet
---
You are a hackathon planner. Break down the requested feature into:
1. Clear implementation steps
2. Files that need to be created/modified
3. Dependencies needed
4. Estimated complexity (low/medium/high)
Keep plans concise — this is a hackathon, speed matters.
```

**`rag-builder.md`** — for RAG pipeline work:
```markdown
---
name: rag-builder
description: Builds RAG pipelines with LangChain
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
model: sonnet
---
You specialize in building RAG pipelines. You use:
- LangChain for document loading and chunking
- LangGraph for orchestrating retrieval agents
- ChromaDB or Pinecone for vector storage
- Proper chunking strategies (RecursiveCharacterTextSplitter)
Always create Pydantic schemas for inputs/outputs.
```

**`frontend-dev.md`** — for UI work:
```markdown
---
name: frontend-dev
description: Builds React/Next.js frontend components
tools: ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
model: sonnet
---
You build Next.js 15 App Router frontends with TypeScript and Tailwind.
- Use server components by default, client components only when needed
- Use shadcn/ui for UI components
- Keep components small and composable
- Always handle loading and error states
```

### 3.6 Create Key Commands

Create `~/.claude/commands/` with hackathon shortcuts:

**`plan.md`:**
```markdown
Break down this feature into implementation steps. Consider the existing codebase structure and output a clear, ordered plan. This is a hackathon — optimize for speed, not perfection.

Feature: $ARGUMENTS
```

**`scaffold.md`:**
```markdown
Create the file structure and boilerplate for: $ARGUMENTS
Include proper types, imports, and placeholder implementations. Wire everything up so it compiles.
```

**`demo-prep.md`:**
```markdown
Review the current project state and:
1. List all working features
2. Identify any broken/incomplete features
3. Suggest the best demo flow
4. Flag any crashes or error states to avoid during demo
5. Create a demo script/talking points
```

---

## Part 4: Hackathon Day — Optimal Workflow

### 4.1 The Two-Instance Kickoff (First 30 Minutes)

Open **two terminals** side by side:

**Terminal 1 (Left) — Scaffolding Agent:**
```bash
cd hackathon-project
claude
```
This instance lays down the foundation:
- Project structure
- Package installations
- Config files
- Database schema
- Basic routing

**Terminal 2 (Right) — Research Agent:**
```bash
cd hackathon-project
claude
```
This instance handles research:
- Creates the detailed PRD/plan
- Looks up LangGraph documentation
- Finds the right LangChain components
- Researches RAG best practices for your use case

Use `/rename scaffolding` and `/rename research` to label them.

### 4.2 The Build Phase

Once scaffolding is done, shift to focused building:

**Main terminal:** Code changes — one feature at a time
**Second terminal:** Questions, documentation lookups, debugging help

**Workflow per feature:**
1. `/plan "Add RAG pipeline with PDF ingestion"` — get a plan
2. Review the plan, adjust if needed
3. Let Claude implement it
4. Test it manually
5. `/compact` — clear context before next feature
6. Move to next feature

### 4.3 Parallelization (When You're Comfortable)

If you need to work on frontend and backend simultaneously:

```bash
# Create git worktrees
git worktree add ../hackathon-frontend frontend-branch
git worktree add ../hackathon-backend backend-branch

# Terminal 1: Frontend
cd ../hackathon-frontend && claude

# Terminal 2: Backend
cd ../hackathon-backend && claude
```

**Rules for parallel work:**
- Each instance works on **non-overlapping files**
- Frontend and backend are naturally orthogonal — perfect for parallel
- Use `/rename` on each chat
- Keep to **2–3 instances max** — more creates mental overhead faster than it creates output
- Merge branches carefully at the end

### 4.4 Key Slash Commands During the Hackathon

| Command | When to Use |
|---------|-------------|
| `/plan "feature"` | Before starting any new feature |
| `/compact` | Between features, after debugging |
| `/clear` | When switching to completely unrelated work |
| `/model opus` | Complex architecture or persistent bugs |
| `/model sonnet` | Everything else (default) |
| `/cost` | Check token usage periodically |
| `/fork` | Need to explore something without polluting main context |

### 4.5 Building Your Agentic AI System

For your specific hackathon (multi-agent, RAG, LangGraph), here's the recommended build order:

**Hour 1–2: Foundation**
```
1. Scaffold project (frontend + backend)
2. Set up FastAPI with basic health endpoint
3. Set up Next.js with basic page
4. Connect frontend to backend
5. Set up Supabase/database
```

**Hour 2–4: Core AI Pipeline**
```
1. Set up LangGraph state machine for your agent flow
2. Build RAG pipeline:
   - Document loader (PDF/web)
   - Chunking strategy
   - Vector store (ChromaDB for local dev, Pinecone for prod)
   - Retrieval chain
3. Create individual agent nodes
4. Wire agents together in LangGraph
5. Expose via FastAPI endpoints
```

**Hour 4–6: Frontend & Integration**
```
1. Build chat/interaction UI
2. Connect to backend API
3. Add streaming responses (SSE or WebSocket)
4. Add agent visualization (show which agent is working)
5. Polish UI with Tailwind + shadcn
```

**Final Hour: Demo Prep**
```
1. Run /demo-prep command
2. Fix any broken flows
3. Prepare seed data / demo scenarios
4. Write a compelling README
5. Practice the demo flow
```

### 4.6 Prompting Tips for Speed

**Be specific and front-load context:**
```
# BAD (vague, will need follow-ups)
"Add a RAG pipeline"

# GOOD (specific, one-shot)
"Create a RAG pipeline in backend/agents/rag.py using LangChain.
- Use RecursiveCharacterTextSplitter with chunk_size=1000
- Use ChromaDB as vector store (persist to ./chroma_db)
- Use OpenAI embeddings (text-embedding-3-small)
- Create a retrieval chain that takes a query string and returns
  top 5 relevant chunks with source metadata
- Add a FastAPI endpoint POST /api/rag/query
- Include Pydantic request/response models
- Add basic error handling"
```

**Use `@file` references instead of pasting code:**
```
"Look at @backend/agents/orchestrator.py and add a new node
for the summarization agent. Follow the same pattern as the
existing research_node."
```

**Chain commands in a single prompt:**
```
"Run /tdd for the RAG pipeline, then /code-review the result"
```

---

## Part 5: Critical Survival Tips

### 5.1 Context Window Management

Your 200k context window is precious. Things that eat it:
- Enabled MCPs (each tool description costs tokens)
- Long files Claude has to read repeatedly
- Accumulated conversation history

**Keep healthy by:**
- Disabling unused MCPs
- Using `/compact` at logical breakpoints
- Keeping files under 300 lines
- Using `/clear` between unrelated tasks

### 5.2 When Claude Gets Stuck

1. **Don't repeat the same prompt** — rephrase with more context
2. **Switch to Opus** for complex problems: `/model opus`
3. **Break the problem down** — ask Claude to solve a smaller piece first
4. **Use `/fork`** to explore a different approach without losing your main context
5. **Check the docs yourself** — sometimes it's faster to look at the LangGraph/LangChain docs directly and then tell Claude what to do

### 5.3 Copilot Pro + Claude Code Synergy

Use them for different things:
- **Copilot:** Inline autocomplete, boilerplate, repetitive patterns
- **Claude Code:** Multi-file refactoring, architectural decisions, debugging, writing tests, git operations

They're complementary, not competing. Copilot speeds up your typing; Claude Code handles the complex reasoning.

### 5.4 Git Hygiene

```bash
# Commit frequently (every working feature)
git add -A && git commit -m "feat: add RAG pipeline with PDF ingestion"

# Use branches for risky changes
git checkout -b experiment/new-agent-flow

# If it works, merge. If not, just switch back.
git checkout main
```

Claude Code can handle git for you — just ask: "Commit what we just did with a descriptive message."

### 5.5 Emergency Fallbacks

If you **run out of Claude Code tokens:**
- Switch to claude.ai for chat-based help (shares the same pool, but lighter)
- Use Copilot Pro more heavily for code generation
- Use the Anthropic API with a Console account (pay-per-token, separate from Pro limits)
- Have backup API credits loaded in Console beforehand, just in case

If Claude Code **gives bad output:**
- `/rewind` to go back to a previous state
- Use `/checkpoints` for file-level undo
- `git stash` or `git checkout -- .` to revert changes
- Don't rage-prompt — take a breath, rephrase clearly

---

## Quick Reference Card

```
INSTALL:     curl -fsSL https://claude.ai/install.sh | sh
START:       claude
DIAGNOSE:    claude doctor
CLEAR:       /clear (free context reset)
COMPACT:     /compact (summarize and compress context)
COST:        /cost (check token usage)
MODEL:       /model sonnet | /model opus | /model haiku
FORK:        /fork (branch conversation)
RENAME:      /rename <name>
REWIND:      /rewind (undo to previous state)
PLAN:        /plan "feature description"
HELP:        /help
```

---

## Pre-Hackathon Checklist

- [ ] Claude Code installed and authenticated with Pro account
- [ ] Node.js 18+ installed (for MCPs)
- [ ] `claude doctor` runs clean
- [ ] `~/.claude/settings.json` has token optimization settings
- [ ] Everything Claude Code plugin installed
- [ ] Rules copied to `~/.claude/rules/`
- [ ] Key MCPs configured (GitHub, web search)
- [ ] Custom agents created (planner, rag-builder, frontend-dev)
- [ ] Custom commands created (plan, scaffold, demo-prep)
- [ ] CLAUDE.md template ready for your project
- [ ] Editor set up (VS Code/Cursor with Copilot Pro)
- [ ] Git configured
- [ ] Tested a basic Claude Code session end-to-end
- [ ] Have backup API credits in Anthropic Console (just in case)
- [ ] LangGraph + LangChain docs bookmarked
- [ ] Practiced the two-instance workflow
