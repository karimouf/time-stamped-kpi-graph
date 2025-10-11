# Time-Stamped KPI Graph Builder

A comprehensive system for extracting Key Performance Indicators (KPIs) from structured table data and building time-stamped knowledge graphs with semantic verification.

## 🎯 Overview

This project addresses the challenge of extracting and organizing KPI data from complex table structures found in financial reports, annual reports, and similar documents. It creates a graph-based representation where nodes represent KPIs for specific entities and years, while edges connect the same KPIs across time periods.

### Key Features

- **🔍 Intelligent Table Parsing**: Handles complex multi-level headers and merged column structures
- **📊 KPI Extraction**: Automatically identifies and extracts KPI names, entity keys, values, and temporal information
- **🕐 Time-Stamped Graphs**: Creates nodes with temporal relationships via directed edges
- **🔬 Semantic Verification**: Built-in consistency checking for extracted data quality
- **📈 Visualization Tools**: Multiple visualization options for graph analysis
- **💾 Data Export**: JSON export for further analysis and integration

## 🏗️ Architecture

### Graph Structure

**Nodes** represent individual KPI data points with:
- **KPI Name**: Normalized metric name (e.g., "sales_revenue", "vehicle_sales")
- **Entity Key**: Company/division identifier (e.g., "Audi", "Volkswagen Passenger Cars")
- **Value**: Actual metric value (e.g., "61753 million")
- **Year**: Time period (e.g., 2022)
- **Evidence**: Complete metadata about the source

**Edges** connect temporal relationships:
- Link same KPI-entity combinations across years
- Include year difference metadata
- Enable time-series analysis

### Example Node Structure
```json
{
  "id": "sales_revenue_audi_2022",
  "kpi_name": "sales_revenue",
  "key": "Audi",
  "value": "61753 million",
  "year": 2022,
  "evidence": {
    "table_id": "VW2022_Tc4f352",
    "doc_id": "VW2022",
    "page": 14,
    "section_name": "AUDI (PREMIUM BRAND GROUP)",
    "title": "Audi – Key Figures"
  }
}
```

## 📁 Project Structure

```
time-stamped-kpi-graph/
├── data/
│   ├── tables/
│   │   └── linked_tables.jsonl       # Input table data
│   └── paragraphs/
│       └── linked_paragraphs.jsonl   # Supporting paragraph data
├── kpi_graph_builder.py              # Core extraction logic
├── kpi_visualization.py              # Visualization tools
├── demo.py                           # Complete demonstration
├── requirements.txt                  # Python dependencies
├── kpi_graph_export.json            # Sample exported graph
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karimouf/time-stamped-kpi-graph.git
   cd time-stamped-kpi-graph
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demonstration**
   ```bash
   python demo.py
   ```

### Basic Usage

```python
from kpi_graph_builder import KPIGraphBuilder
from kpi_visualization import KPIGraphVisualizer

# Build the graph
builder = KPIGraphBuilder()
graph = builder.build_graph_from_tables("data/tables/linked_tables.jsonl")

# Get statistics
stats = builder.get_graph_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Total edges: {stats['total_edges']}")

# Visualize
visualizer = KPIGraphVisualizer(graph)
visualizer.plot_kpi_distribution()
```

## 🔧 Core Components

### KPIExtractor
Handles the intelligent parsing of table structures:
- **Header Analysis**: Processes complex multi-level headers
- **Year Detection**: Extracts temporal information from merged headers
- **KPI Normalization**: Standardizes KPI names for consistency
- **Value Cleaning**: Handles missing data markers and formatting

### KPIGraphBuilder
Constructs the time-stamped graph:
- **Node Creation**: Generates unique nodes with complete metadata
- **Edge Generation**: Creates temporal connections between related KPIs
- **Graph Management**: Uses NetworkX for efficient graph operations

### SemanticVerifier
Ensures data quality and consistency:
- **Format Validation**: Checks value formats against KPI types
- **Metadata Consistency**: Verifies temporal alignment
- **Quality Reporting**: Provides detailed validation statistics

### KPIGraphVisualizer
Provides multiple visualization options:
- **Network Graphs**: Interactive node-edge visualizations
- **Timeline Plots**: KPI trends across time periods
- **Distribution Analysis**: Statistical overviews
- **Data Export**: JSON and CSV export capabilities

## 📊 Data Format

### Input Format (JSONL)
The system expects JSONL files with table structures:

```json
{
  "table_id": "VW2022_Td85de9",
  "doc_id": "VW2022",
  "year": 2022,
  "page": 4,
  "section_name": "KEY FIGURES BY BRAND",
  "title": "Key Figures by brand and business field",
  "headers": [["", "SALES REVENUE", "SALES REVENUE"], ["Units", "2022", "2021"]],
  "merged_headers": ["Units", "SALES REVENUE 2022", "SALES REVENUE 2021"],
  "rows": [["Audi", "61753", "55914"], ["ŠKODA", "21023", "17743"]],
  "stub_col": ["Audi", "ŠKODA"]
}
```

### Output Format (Graph)
NetworkX DiGraph with rich node and edge attributes suitable for:
- Time-series analysis
- Comparative studies
- Trend identification
- Anomaly detection

## 🎨 Visualization Examples

### 1. KPI Distribution Analysis
```python
visualizer.plot_kpi_distribution()
```
Shows distribution of KPIs, entities, and temporal coverage.

### 2. Timeline Visualization
```python
visualizer.plot_kpi_timeline("sales_revenue", ["Audi", "ŠKODA"])
```
Displays KPI trends over time for specific entities.

### 3. Network Graph
```python
visualizer.plot_network_graph(kpi_filter="sales_revenue")
```
Interactive network showing KPI relationships.

## 🔍 Advanced Features

### Semantic Consistency Checking
```python
from kpi_graph_builder import SemanticVerifier

verifier = SemanticVerifier()
results = verifier.verify_graph(graph)
print(f"Consistency rate: {results['consistency_rate']:.1%}")
```

### Custom KPI Patterns
```python
extractor = KPIExtractor()
extractor.kpi_patterns['custom_metric'] = ['revenue', 'income', 'earnings']
```

### Graph Export
```python
visualizer.export_graph_data("my_graph_export.json")
```

## 📈 Use Cases

### Financial Analysis
- **Multi-year Performance Tracking**: Compare KPIs across time periods
- **Entity Benchmarking**: Analyze performance across different business units
- **Trend Analysis**: Identify growth patterns and anomalies

### Business Intelligence
- **Automated Reporting**: Extract KPIs from structured reports
- **Data Integration**: Combine data from multiple sources
- **Quality Assurance**: Verify data consistency across documents

### Research Applications
- **Longitudinal Studies**: Track metrics over extended periods
- **Comparative Analysis**: Study performance across entities
- **Data Mining**: Discover patterns in business metrics

## 🛠️ Configuration

### KPI Pattern Customization
Modify `kpi_patterns` in `KPIExtractor` to add domain-specific metrics:

```python
kpi_patterns = {
    'sales_revenue': ['sales revenue', 'revenue', 'turnover'],
    'profit_margin': ['profit margin', 'operating margin', 'ebitda margin'],
    'custom_kpi': ['your', 'custom', 'patterns']
}
```

### Visualization Settings
Customize visualization parameters:

```python
visualizer.plot_network_graph(
    kpi_filter="sales_revenue",
    max_nodes=50,
    layout='spring'  # or 'circular', 'hierarchical'
)
```

## 🔧 Dependencies

- **networkx**: Graph construction and analysis
- **matplotlib**: Basic plotting and visualization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **seaborn**: Statistical visualizations

## 📝 Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
The project follows PEP 8 guidelines. Use `black` for formatting:
```bash
black *.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Missing Data Files**
```
FileNotFoundError: data/tables/linked_tables.jsonl
```
Ensure data files are in the correct directory structure.

**Import Errors**
```
ModuleNotFoundError: No module named 'networkx'
```
Install all requirements: `pip install -r requirements.txt`

**Visualization Issues**
```
Cannot display plots
```
For headless environments, use `matplotlib.use('Agg')` before importing pyplot.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for automotive industry financial data analysis
- Inspired by knowledge graph construction methodologies
- Uses NetworkX for efficient graph operations
- Visualization powered by matplotlib and seaborn

## 📞 Contact

For questions, issues, or contributions:
- GitHub: [@karimouf](https://github.com/karimouf)
- Project: [time-stamped-kpi-graph](https://github.com/karimouf/time-stamped-kpi-graph)

---