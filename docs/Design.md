# Design & Styling Guidelines

## 1. Code Formatting
- Follow PEP 8 guidelines.
- Use Python 3.10+ typing syntax natively (`|` instead of `Union`, standard collections `list`, `dict` instead of `typing.List`).
- Max line length of 100 characters.

## 2. API Design
- **FastAPI Standards:** Use Pydantic V2 models for all request/response schemas. Include descriptive titles and field descriptions.
- **Explainability:** When returning MoE predictions, always include an `attribution_dict` (e.g., `{"eager_pct": 0.6, "revisit_pct": 0.4}`) to provide transparency to the UI/operator.

## 3. Terminal & Visual Output
- **Environment Rendering:** The `RFScanEnv.render()` method should output a clean, visually distinct terminal representation (e.g., using ASCII bars or color-coded blocks) of the current spectrum state and recent agent actions.
- **Plotting:** Standardize all matplotlib/seaborn plots generated during evaluation:
  - High DPI (300) for PDFs.
  - Proper axis labeling with units (e.g., "Time ($\mu$s)", "Frequency (MHz)").
  - Use accessible color palettes (e.g., `viridis` or Seaborn's `colorblind`).

## 4. Documentation Styling (Markdown)
- Use standard Markdown headings.
- Include mermaid diagrams or ASCII art where applicable to describe complex flows like Dueling DQN streams or MoE fusion.
- Use explicit code blocks with appropriate syntax highlighting (e.g., ```python, ```bash, ```yaml).
