# Papers MCP

An MCP (Model Context Protocol) server for searching and retrieving academic papers. Designed for AI agent usage with optimized tool signatures and descriptions.

## Features

- **Paper Search**: Search for academic papers by relevancy query using Semantic Scholar API
- **Paper Details**: Retrieve detailed information about a specific paper by ID

## Installation

```bash
uv sync --all-extras
```

## Usage

### Running the Server

```bash
make serve
```

Or directly:

```bash
uv run papers-mcp
```

### Development

```bash
# Run tests
make test

# Run linting
make lint

# Run type checking
make typecheck

# Format code
make format

# Run all checks
make check
```

## MCP Tools

### `search_papers`

Search for academic papers based on a query. Returns relevant papers with titles, authors, and abstracts.

### `get_paper`

Retrieve detailed information about a specific paper by its Semantic Scholar ID.

## Architecture

This project follows functional domain-driven design principles:

- **Domain Types**: Immutable dataclasses representing papers and search results
- **Services**: Pure functions for API interactions
- **Server**: FastMCP server exposing tools for AI agents

## License

MIT
