"""CLI entry point (instruction.md Section 7.8 / 9.6).

Usage:
    python -m src.cli ingest --path data/raw_pdfs/<file>.pdf
    python -m src.cli ask --query "..."
"""

from __future__ import annotations

import typer

from src.generation import generate_answer
from src.ingest import ingest as ingest_pdf
from src.retrieval import retrieve

app = typer.Typer(help="Hierarchical RAG analyzer CLI")


@app.command()
def ingest(path: str = typer.Option(..., "--path", help="Path to the PDF to ingest")) -> None:
    """Ingest a PDF into the parent store and Chroma vector index."""
    ingest_pdf(path)


@app.command()
def ask(query: str = typer.Option(..., "--query", help="Question to ask")) -> None:
    """Retrieve relevant chunks and generate a grounded, cited answer."""
    chunks = retrieve(query)
    result = generate_answer(query, chunks)
    typer.echo("\n" + result["answer"] + "\n")
    if result["citations"]:
        typer.echo("Citations:")
        for c in result["citations"]:
            typer.echo(f"  - {c['doc_name']}, page {c['source_page']}")
    else:
        typer.echo("Citations: none")


if __name__ == "__main__":
    app()
