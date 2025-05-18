#!/usr/bin/env python3
"""
Command-line interface for Java CSS Optimizer.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from java_css_optimizer.analyzer import JavaAnalyzer
from java_css_optimizer.optimizer import JavaOptimizer
from java_css_optimizer.reporter import OptimizationReporter
from java_css_optimizer.config import Config

# Initialize Typer app
app = typer.Typer(
    name="java-css-optimizer",
    help="Optimize Java UI code by extracting styles to CSS",
    add_completion=False,
)

# Initialize Rich console
console = Console()

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Java file to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Analyze a Java file for styling patterns."""
    setup_logging(verbose)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing Java file...", total=None)
            
            analyzer = JavaAnalyzer()
            result = analyzer.analyze_file(file)
            
            progress.update(task, completed=True)
            
            if output:
                output.mkdir(parents=True, exist_ok=True)
                result.save(output)
                console.print(f"[green]Analysis saved to {output}[/green]")
            else:
                console.print(result)
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

@app.command()
def optimize(
    directory: Path = typer.Argument(..., help="Directory containing Java files"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    css_name: Optional[str] = typer.Option(None, "--css-name", help="CSS file name"),
    level: int = typer.Option(2, "--level", "-l", help="Optimization level (1-3)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Optimize Java files by extracting styles to CSS."""
    setup_logging(verbose)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing Java files...", total=None)
            
            config = Config(
                optimization_level=level,
                css_output=output,
                css_name=css_name,
            )
            
            optimizer = JavaOptimizer(config)
            result = optimizer.optimize_directory(directory)
            
            progress.update(task, completed=True)
            
            console.print(f"[green]Optimization complete![/green]")
            console.print(result.summary())
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

@app.command()
def report(
    directory: Path = typer.Argument(..., help="Directory to analyze"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("text", "--format", "-f", help="Report format (text/html/json)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Generate an optimization report."""
    setup_logging(verbose)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating report...", total=None)
            
            reporter = OptimizationReporter()
            report = reporter.generate_report(directory, format)
            
            progress.update(task, completed=True)
            
            if output:
                output.write_text(report)
                console.print(f"[green]Report saved to {output}[/green]")
            else:
                console.print(report)
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

@app.command()
def suggest(
    file: Path = typer.Argument(..., help="Java file to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Show optimization suggestions for a Java file."""
    setup_logging(verbose)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing file...", total=None)
            
            analyzer = JavaAnalyzer()
            suggestions = analyzer.get_suggestions(file)
            
            progress.update(task, completed=True)
            
            console.print("\n[bold]Optimization Suggestions:[/bold]")
            for suggestion in suggestions:
                console.print(f"\n[blue]• {suggestion.description}[/blue]")
                console.print(f"  [yellow]Location:[/yellow] {suggestion.location}")
                console.print(f"  [yellow]Impact:[/yellow] {suggestion.impact}")
                if suggestion.example:
                    console.print(f"  [yellow]Example:[/yellow]\n{suggestion.example}")
                
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

def main() -> None:
    """Main entry point."""
    app() 