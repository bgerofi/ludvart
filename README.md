# Ludvart

<!--
Are you tired of talking to your favorite LLM in a dedicated chat window and
copying text back and forth between it and your terminal? Have you ever wished
for an AI agent that could appear out of nowhere and help with whatever you're
working on? Have you had enough of setting up custom coding tools just to handle
everyday CLI tasks? Then you're in the right place.
-->


**ludvart** is an AI agent that sits transparently on the character stream between
you and any terminal-based program you're running. It sees exactly what you see, 
and it's ready to help with whatever you need, wherever you need it.

**ludvart** operates at the pseudo-terminal (PTY) layer rather than inside any
particular application, thus it integrates **seamlessly with any terminal and any
program**: arbitrary shells, full-screen TUI apps (`htop`, `vim`, `claude`), and
REPLs all work unchanged. There is nothing to configure per-app; if it runs in a
terminal, ludvart can drive it.

ludvart is **host transparent**, it processes the PTY byte stream and it
travels with you across `ssh` hops, nested `tmux`/`screen` sessions,
even interactive jobs on a supercomputer.
It keeps working on the far side without needing any agent or API key installed
on the remote host, or the right proxy settings configured.
Your session, wherever it goes, carries the agent along.

You are in control by default, **ludvart** does not disturb you, but once you summon
the agent, it can:

- **Run commands** on your behalf (and read back their output).
- **Control interactive applications** by sending real keystrokes, edit in
  `vim`, page through `less`, drive a Python REPL, and so on.
- **Focus on specific parts of the screen, including scrollback history**, so it
  can reason about exactly what you have been looking at.
- Through enhanced **[helpers](#assistant-helpers)** it can also acomplish more
  complex tasks, such as coding or debugging, issue resolution or triaging, etc.

## What Ludvart is not?

- Ludvart is **not an MCP service or a plugin** for extending other harnesses. 
  It does not exist to hand tools or context to a separate AI harness. It *is* the
  agent, and it drives any harness by itself, at the PTY layer, sending real
  keystrokes and reading the real screen. There is no host application it needs
  to be embedded in and nothing to register on the far side.
- Ludvart is **not a terminal emulator with AI bolted on**. It does not implement a
  terminal, swap you onto a different shell, or ask you to adopt a new one. It
  launches your own `$SHELL` (or the command you give it) and runs *inside*
  whatever terminal you already use (xterm, iTerm, Alacritty, ghostty, Windows Terminal,
  a `tmux`/`screen` pane, etc.), relaying the byte stream transparently.
  Your terminal, keybindings, and workflow stay exactly as they were.
- Ludvart is **not a shell modification**. It does not patch, wrap, or replace your
  shell, and it is not tied to any particular one. It works with any shell and any
  CLI tool: a database prompt, a code editor, a debugger, a REPL,
  or whatever else you launch.


## Getting Started

To set up the ludvart environment, run:

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a local `.venv`, installs ludvart, and also installs the
optional **LiteLLM gateway** (`litellm[proxy]`) that enables the **GitHub
Copilot** backend. To start ludvart, type:

```bash
ludvart
```

The first run walks you through registering a model — see
[LLM configuration](#llm-configuration). After that, press **Ctrl-O** to open
the AI panel and run:

```
/init_helpers
```

This installs `ludvart_helper` on the machine you are currently driving. The
agent's higher-level abilities — reliable file read/edit/search, patching,
running commands with a real exit code — are built on it and are *not*
available until it is installed. Because ludvart travels with your session,
run `/init_helpers` again on the far side of an `ssh` hop the first time you
want those abilities there. See [Assistant helpers](#assistant-helpers).

## LLM configuration

ludvart keeps every model you register in `~/.ludvart/models.json` — an array of
registrations, each holding a provider, endpoint URL, API key, and model name.
One entry is marked *active* (the model currently in use).

The first time you run `ludvart` with an empty registry, it walks you through a
short **interactive setup** that registers your first model:

```
Select the API endpoint type:
  1) OpenAI
  2) Anthropic
  3) Google
  4) Custom (OpenAI-compatible)
  5) GitHub Copilot (via local LiteLLM gateway)
```

For providers 1–4 it asks for the endpoint URL, API key (hidden), and model
name. Option 5 (GitHub Copilot) runs GitHub's device-flow authorization and then
lets you pick from the models your account can use — see
[GitHub Copilot](#more-on-github-copilot-via-litellm) below.

Once at least one model is registered, you can add and switch between models from
the AI panel with the **`/model`** command (see [Managing models](#managing-models)).
At startup ludvart verifies **every** registered model and marks the reachable
ones as *available*.


## Summoning the agent

Run your favorite application or stay with your everyday work in shell. Whenever
you need help, press **Ctrl-O** to open (or close) the AI panel — a bottom split where 
you type to the agent while your program keeps running above. Ctrl-O is used because
`screen` (Ctrl-A) and `tmux` (Ctrl-B) leave it alone, so it works even inside
nested sessions.

Inside the panel:

- Type your request and press **Enter** to send it; **Esc** or **Ctrl-O** closes
  the panel.
- The input is a full editor: arrow keys, Home/End, Ctrl-A/E, Ctrl-U/K/W, and
  mouse (bracketed) paste all work.
- **The input is multi-line.** **Alt-Enter** opens a new line, a paste keeps the
  newlines it came with, and Up/Down move within the input before they scroll
  the transcript — so a pasted code snippet or log excerpt keeps its shape.
- **Shift with any arrow** (or Shift-Home / Shift-End) extends a selection,
  shown in reverse video; typing or Backspace then replaces or deletes the whole
  block. Some terminals never report Shift, so **Ctrl-Space** sets a mark that
  makes plain movement select instead; press it again to cancel. (PuTTY drops
  the modifier by default: set Terminal -> Keyboard -> "Shift/Ctrl/Alt with the
  arrow keys" to "xterm-style bitmap" to get the shifted arrows themselves.)
- **PageUp / PageDown** scroll the conversation by a page; Up/Down scroll it a
  line at a time once the cursor is at the edge of the input.

## Assistant tools

Once an LLM provider is configured, the AI agent can act inside your terminal
using a set of tools. The primary one is **`inject_input`**.

### `inject_input`

`inject_input` types characters into your terminal exactly as if you had pressed
the keys yourself. Whatever program is currently in the foreground receives the
input, which makes the tool useful in two ways:

- **Run shell commands** on your behalf — e.g. `ls`, `cat`, checking status, or
  installing packages. Submitting the input (pressing Enter) executes them.
- **Send keystrokes to interactive programs** — control characters and TUI
  navigation for editors, pagers, and REPLs (`vim`, `less`, a Python shell, etc.).

Because ludvart operates at the PTY layer, injected input flows through the same
byte stream as your own keystrokes, so it works with plain shells, full-screen
ncurses apps, and nested remote sessions alike.

| Field    | Description                                                        |
|----------|--------------------------------------------------------------------|
| `text`   | The exact characters to type (may include control characters).     |
| `submit` | If `true`, press Enter after the text to execute it. Default `false`. |

The result of the injected input appears on the terminal screen, which the agent
can then read back and act on.

### `capture_screen_history`

`capture_screen_history` lets the agent focus on a specific part of the screen,
including content that has scrolled off the top. It reads the full logical buffer
(scrollback plus the current viewport) and returns a slice of it, so the agent
can go back and inspect earlier output rather than only the visible rows.

| Field    | Description                                                        |
|----------|--------------------------------------------------------------------|
| `offset` | Where to start; a negative value counts lines above the current position. |
| `length` | How many lines to return.                                          |

Because the injected-input result and the screen history both come from ludvart's
own `pyte` model of the terminal, the agent always sees exactly what you see —
across plain shells, full-screen apps, and remote sessions alike.

## Assistant helpers

Tools are the agent's built-in, in-process primitives — `inject_input` and
`capture_screen_history` — and they are the *only* channel through which the
agent touches your machine. Everything the agent does ultimately flows through
them. **Helpers** are a complementary mechanism layered *on top of* those
tools.

Because ludvart works purely at the PTY layer, the harness has no direct
filesystem or exec access to the (possibly remote) box it is driving — it only
sees the terminal. To work reliably at a higher level, the agent uses a small,
dependency-free helper program, `ludvart_helper`, under `~/.ludvart/bin/` on that
machine. The canonical helper ships *with* ludvart as a version-pinned,
checksummed copy.

The agent does not call helpers directly the way it calls a tool; it *runs*
them by typing a shell command through `inject_input`. `ludvart_helper` exposes
subcommands for the file operations that are awkward to do safely over a raw
terminal — `read`, `write`, `append`, `replace`, `replace-range`, `structured-patch`, `search`,
`run`, and `info`. Every content payload is passed as base64 and every result is
sentinel-framed with a real exit code, so edits are immune to quoting, newline,
and escape corruption, and success is read from a reliable status rather than
guessed from screen text.

You install or repair it with the `/init_helpers` panel command, which you run
once per machine before the agent's file work is available (see
[Getting Started](#getting-started)). This is deterministic and does *not*
involve the model: ludvart injects a short, self-contained shell command that
compares the on-disk copy against the bundled version by checksum and rewrites
it only if it is missing, outdated, or modified.

### Tools vs. helpers

| | Tools | Helpers |
|---|-------|---------|
| **What** | Built-in agent primitives | A version-pinned script bundled with ludvart |
| **Where they run** | Inside the ludvart process (Python) | On the target machine, under `~/.ludvart/bin/` |
| **How invoked** | Called directly by the model | Run via `inject_input` (typed as shell commands) |
| **Availability** | Always present | Optional; installed/repaired via `/init_helpers` |
| **Lifetime** | Live for the process | Persist across sessions |
| **Purpose** | The only way the agent acts at all | Make file read/edit/search reliable and corruption-proof |


## Miscelanous

### Managing models

Inside the AI panel, `/model` manages the registry:

- `/model list` — list all registered models; the one in use and the ones that
  passed verification are marked.
- `/model add` — register a new model through guided prompts (endpoint type,
  URL, API key, model name; the key is hidden as you type). For GitHub Copilot it
  lists the models your subscription can use. The new model is verified before it
  is saved and does not become the active one.
- `/model use <n>|<name>` — switch to another registered model by its list number
  or a substring of its name (only *available* models can be selected).
- `/model remove <n>|<name>` — unregister a model (not the one currently in use).

Switching to (or adding) a GitHub Copilot model launches its local gateway on
demand. ludvart runs **one gateway at a time** — for the active Copilot model —
and tears it down when you switch away or exit.


### More on GitHub Copilot (via LiteLLM)

ludvart can use **GitHub Copilot** as an OpenAI-compatible backend by spawning a
local [LiteLLM](https://github.com/BerriAI/litellm) proxy that fronts LiteLLM's
`github_copilot/` provider. ludvart then talks to that proxy with its normal
OpenAI-compatible client — no endpoint URL or API key to manage.

Requirements and behavior:

- An **active paid GitHub Copilot subscription**.
- The gateway (`litellm[proxy]`) must be installed — `setup.sh` does this, or
  run `uv pip install 'litellm[proxy]'`.
- Authorization uses **GitHub's OAuth device flow**: ludvart prints a URL and a
  one-time code; you open the URL, enter the code, and approve access. The
  credentials are cached under `~/.config/litellm/github_copilot` and reused on
  later runs, so the gateway starts non-interactively afterwards. (This uses
  GitHub's own OAuth, not `~/.netrc`.)

The easiest way to add Copilot is the first-run wizard (option 5) or `/model add`
→ GitHub Copilot, which lists the models your account can use (Copilot uses its
own model ids, e.g. `gpt-4o`, `claude-opus-4.8`). The chosen model is stored as a
registration in `models.json`:

```jsonc
// ~/.ludvart/models.json (one entry)
{ "provider": "copilot", "model": "gpt-5.3-codex", "active": true }
```

You can register **several** Copilot models. ludvart spawns the local LiteLLM
gateway on loopback only for the Copilot model that is currently active, points
its OpenAI-compatible client at it, and shuts it down when you switch away or
exit. The first device-flow authorization must be done in a terminal (the
first-run wizard); once authorized, adding Copilot models from the panel works
without re-authorizing.

### Timeouts and retries

Each request waits up to **`LUDVART_LLM_TIMEOUT`** seconds (default `120`). This
is applied as the *read* timeout, with a short independent connect timeout, so a
slow model that takes a while to start streaming (common for reasoning models
behind the Copilot gateway) is not mistaken for an unreachable endpoint. On a
transient failure — a timeout, a dropped connection, a rate limit, or a `5xx`
response — ludvart retries up to **`LUDVART_LLM_MAX_RETRIES`** times (default `2`,
with exponential backoff). Both settings are read from the environment.

### Context window

ludvart tracks how much of the model's context window a conversation uses (shown
as a `[NN%]` badge in the panel, and used to trigger automatic summarization).
It learns the window size from the provider's API when possible; otherwise it
falls back to a small table of known models.

On first run that table is written to **`~/.ludvart/context_windows.json`** — a
self-documented JSON file you can edit. Each key is matched as a
case-insensitive substring of the model id and the first match wins, so keep the
most specific ids first:

```jsonc
{
  "claude-opus-4": 1000000,
  "gpt-5": 400000,
  "gpt-4o": 128000,
  "my-local-model": 32768
}
```

ludvart re-reads the file whenever it changes, so edits take effect on the next
request. Delete the file to regenerate the defaults. For a one-off override you
can instead set `<PROVIDER>_CONTEXT_WINDOW` (e.g. `CUSTOM_CONTEXT_WINDOW`), which
always wins.




## Related projects

Other projects put an AI agent near the terminal, but they fall into two camps
that are each distinct from Ludvart.

**Headless drivers** — the *agent* spawns and owns a session and drives it
programmatically; the human is out of the loop and reviews the result:

- [agent-tty](https://github.com/coder/agent-tty) (coder) — hands a real,
  long-lived PTY to an agent and records reviewable proof (text snapshots, PNG
  screenshots, WebM video, asciicast) via a Ghostty renderer.
- [agent-terminal](https://github.com/jasonkneen/agent-terminal) (jasonkneen) —
  a `node-pty` wrapper exposed as an MCP server; an external agent reads the
  ASCII buffer and sends keys.
- [pilotty](https://github.com/msmps/pilotty) (msmps) — daemon-managed headless
  PTY sessions with VT100 emulation, snapshots, and detected UI elements.

**In-band assistants** — a human drives and the AI rides along in the *live*
session:

- [Butterfish](https://github.com/bakks/butterfish) (bakks) — the closest on UX.
  Wraps your shell in a PTY; you prompt inline (capital letter to ask, `!` for
  agent mode, `@` for a one-shot command) and the AI sees your shell history. It
  reasons over shell command history rather than the rendered screen, and is
  tied to the local `bash`/`zsh`.
- [TmuxAI](https://github.com/alvinunreal/tmuxai) (alvinunreal) — the closest in
  philosophy ("a colleague sitting next to you"). Reads all your tmux panes in
  real time via a chat/exec pane split, but requires tmux.
- [AIShell](https://github.com/changjonathanc/aishell) (changjonathanc) — a
  transparent shell wrapper that captures screen content as AI context, but
  offers help commands rather than an agentic loop.

Further out: [Warp](https://www.warp.dev/) and Microsoft's
[Intelligent Terminal](https://devblogs.microsoft.com/commandline/announcing-intelligent-terminal-version-0-1/)
are full terminal-emulator replacements — the thing ludvart deliberately is *not*
(see [What Ludvart is not?](#what-ludvart-is-not)).

### How Ludvart compares

| Dimension | Ludvart | Butterfish | TmuxAI | AIShell | Headless drivers |
|---|:---:|:---:|:---:|:---:|:---:|
| In-band, human-driven | ✅ | ✅ | ✅ | ✅ | ✗ |
| Full terminal- & program-indepence (i.e., nothing to modify) | ✅ | ✗ (its shell) | ✗ (tmux) | ✗ (its shell) | ✗ (program reachable only through their API/CLI) |
| Works at the raw PTY layer (no shell/tmux/emulator dependency) | ✅ | shell-wrapper | needs tmux | shell-wrapper | spawns its own PTY |
| Reasons about the rendered screen **and** scrollback (not just shell history) | ✅ | ✗ | ✅ | partial | ✅ |
| Drives arbitrary full-screen TUIs (`vim`, `htop`, `claude`) | ✅ | ✗ | ✅ | ✗ | ✅ |
| Host-transparent across `ssh`, nothing installed remotely | ✅ | local shell only | tmux-side | local only | ✗ |
| Resizable in-terminal agent panel + conversation sessions | ✅ | inline shell | chat pane | ✗ | N/A |
| Agentic tool-calling loop | ✅ | ✅ | ✅ | ✗ | driven externally |