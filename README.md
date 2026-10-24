# tg-downloader

A Telegram bot for downloading videos

## Features


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tmwrk34/tg-downloader.git
   cd tg-downloader
   ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):

   ```bash
   uv sync
   ```

3. Create a `.env` file from example:

   ```bash
   cp .env.example .env
   # Edit .env with your BOT_TOKEN and ALLOWED_CHAT_IDS
   ```

4. Run the bot:

   ```bash
   uv run python bot.py
   ```

## Docker Deployment

To run the bot in an isolated Docker container with automated restarts:

1. Create and configure `.env`:

   ```bash
   cp .env.example .env
   # Edit .env with your BOT_TOKEN
   chmod 600 .env
   ```

2. Start the container with Docker Compose:

   ```bash
   docker compose up -d --build
   ```

3. View live logs:

   ```bash
   docker compose logs -f
   ```

4. Stop the container:

   ```bash
   docker compose down
   ```

### Automated CD with GitHub Self-Hosted Runner

To automatically deploy new releases on every push/merge to `main`:

1. On GitHub, navigate to **Settings** ➡️ **Actions** ➡️ **Runners** ➡️ **New self-hosted runner**.
2. Select **Linux** and your architecture (**ARM64** for Raspberry Pi / Oracle Ampere).
3. Follow GitHub's instructions to install the runner on the host.
4. Give the runner user permissions to run Docker:
   ```bash
   sudo usermod -aG docker $USER
   ```
5. Install and start the runner as a background systemd service:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```

Now every push to `main` will automatically trigger `.github/workflows/deploy.yml` and deploy the updated containers.

## Getting a Bot Token

1. Open Telegram and find `@BotFather`
2. Send `/newbot`
3. Follow the instructions
4. Copy the token into `.env`

## Status
Under active development


## Development & Contributing

### Git Hooks Setup

To enforce commit message and branch naming conventions locally, enable the included Git hooks:

```bash
git config core.hooksPath .githooks
# or run the setup script
./.githooks/install.sh
```

### Branch Naming Convention

All branches must follow the pattern:
`<type>/<issue-number>-<description>` or `<type>/<description>`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `hotfix`.

Examples:
- `feat/1-whitelist-middleware`
- `fix/10-audio-extraction`
- `ci/github-actions-setup`

### Commit Message Convention

Commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
`<type>(<scope>): <subject>` or `<type>: <subject>`

Examples:
- `feat(security): implement chat ID whitelist middleware`
- `fix(downloader): handle timeout when fetching metadata`
- `ci: add github actions lint and test workflow`

### Linting & Testing

```bash
# Check code with Ruff
uv run ruff check .

# Format check with Ruff
uv run ruff format --check .

# Run test suite
uv run pytest
```
