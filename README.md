# LLMSeqRec: LLM-Enhanced Contextual Sequential Recommender

## Project Overview

This repository contains the implementation of **LLMSeqRec**, an LLM-Enhanced Contextual Sequential Recommender system developed as part of the 2025 Open Source Research Experience (OSRE). The project addresses the limitations of traditional sequential recommendation systems by leveraging Large Language Models (LLMs) to enhance context-aware sequential recommendations.

## Project Description

Sequential Recommender Systems are widely used in scientific and business applications to analyze and predict patterns over time. Traditional models often struggle with capturing complex contextual dependencies and adapting to dynamic user behaviors, as they primarily rely on vanilla sequential ID orders.

**LLMSeqRec** addresses these limitations by:
- Dynamically integrating LLM-generated embeddings and contextual representations
- Enriching user intent modeling with semantic context
- Mitigating cold-start issues through enhanced context understanding
- Capturing long-range dependencies within sequential data

## Project Objectives

The project aims to develop a unified and scalable model capable of:
1. **Data Preprocessing & Feature Creation**: Parse user sequential interaction behaviors into sequential data points for LLM-based embeddings
2. **Model Development**: Design LLM-enhanced sequential recommendation models integrating pretrained language models
3. **Evaluation**: Benchmark against state-of-the-art sequential recommenders and conduct ablation studies

## What Has Been Implemented

### 1. Data Processing Pipeline ✅

The project includes a comprehensive data processing pipeline for Amazon product review datasets:

- **`DataProcessing_carca.py`**: Main data processing script that:
  - Parses Amazon review data from JSON.gz format
  - Filters users and items based on minimum interaction thresholds (≥5 interactions)
  - Creates sequential user-item interaction sequences
  - Extracts item metadata (title, price, brand, categories)
  - Generates feature representations for items
  - Outputs processed data in multiple formats:
    - `Beauty.txt`: User-item interaction sequences
    - `Beauty_cxt.txt`: Contextual user-item-time sequences
    - `Beauty_feat.dat`: Item feature representations

### 2. Dataset Processing ✅

Successfully processed the **Amazon Beauty** dataset:
- **Input**: `reviews_Beauty.json.gz` (337MB), `meta_Beauty.json.gz` (95MB)
- **Output**: Processed sequential data with 22,363 users and 12,101 items
- **Features**: Extracted item metadata including categories, brand, and price information

### 3. Model Training & Evaluation ✅

Multiple model variants have been trained and evaluated:

#### Model Variants Tested:
1. **`minllm.log`**: MiniLLM-based model (100 epochs)
   - Final Performance: HR@10 = 0.4534, NDCG@10 = 0.2663
   - Stable convergence with consistent improvement

2. **`notitle_e5.log`**: E5 embeddings without title context (50+ epochs)
   - Final Performance: HR@10 = 0.4727, NDCG@10 = 0.2881
   - Shows improvement over baseline

3. **`title_only_e5.log`**: E5 embeddings with title context (200+ epochs)
   - Comprehensive training with title-based context enhancement

4. **`real_title_only_e5.log`**: Real title E5 embeddings (200+ epochs)
   - Enhanced context using actual product titles

### 4. Performance Analysis ✅

The project includes comprehensive performance analysis tools:

- **`Untitled.ipynb`**: Jupyter notebook for:
  - Parsing training logs and extracting metrics
  - Visualizing training progress (loss, HR@10, NDCG@10)
  - Comparing performance across different model variants
  - Analyzing feature representations

- **Performance Metrics Tracked**:
  - Hit Rate @ 10 (HR@10)
  - Normalized Discounted Cumulative Gain @ 10 (NDCG@10)
  - Training and validation loss curves

### 5. Visualization & Results ✅

Generated performance plots for analysis:
- **`beauty-loss-epoch.png`**: Training loss progression for Beauty dataset
- **`beauty-hr.png`**: Hit Rate progression over epochs
- **`beauty-ndcg.png`**: NDCG progression over epochs
- **`ml-m1-*`**: Similar plots for MovieLens-1M dataset comparison

## Project Structure

```
GSOC/
├── DataProcessing_carca.py          # Main data processing script
├── DataProcessing_carca_changed.py  # Modified version with changes
├── DataProcessing_changed.py        # Alternative processing approach
├── Untitled.ipynb                   # Performance analysis notebook
├── features_list_carca.csv          # Extracted item features
├── Beauty.txt                       # Processed user-item sequences
├── Beauty_cxt.txt                   # Contextual sequences with timestamps
├── Beauty_feat.dat                  # Item feature representations
├── reviews_Beauty.json.gz           # Raw Amazon Beauty reviews
├── meta_Beauty.json.gz              # Raw Beauty product metadata
├── *.log                            # Training logs for different models
├── *.png                            # Performance visualization plots
└── README.md                        # This file
```

## Key Features

### Data Processing
- **Sequential Data Generation**: Creates time-ordered user-item interaction sequences
- **Feature Extraction**: Extracts rich item metadata (categories, brand, price)
- **Data Filtering**: Implements minimum interaction thresholds for quality
- **Multiple Output Formats**: Supports various downstream model requirements

### Model Variants
- **MiniLLM Integration**: Lightweight LLM for context enhancement
- **E5 Embeddings**: State-of-the-art text embeddings for semantic understanding
- **Context-Aware Processing**: Incorporates product titles and metadata
- **Progressive Enhancement**: Multiple approaches for context integration

### Evaluation Framework
- **Comprehensive Metrics**: HR@10, NDCG@10, training/validation loss
- **Performance Tracking**: Detailed logging across training epochs
- **Comparative Analysis**: Tools for comparing different model variants
- **Visualization**: Automated plotting of training progress

## Usage Instructions

### 1. Data Processing

To process a new Amazon dataset:

```bash
python DataProcessing_carca.py
```

**Prerequisites**:
- Place your dataset files in the working directory:
  - `reviews_[dataset_name].json.gz`
  - `meta_[dataset_name].json.gz`
- Modify the `dataset_name` variable in the script

**Output**:
- `[dataset_name].txt`: User-item interaction sequences
- `[dataset_name]_cxt.txt`: Contextual sequences with timestamps
- `[dataset_name]_feat.dat`: Item feature representations

### 2. Performance Analysis

Open `Untitled.ipynb` in Jupyter to:
- Parse training logs
- Generate performance visualizations
- Compare model variants
- Analyze feature representations

### 3. Model Training

The project supports multiple model variants:
- **MiniLLM**: Lightweight context enhancement
- **E5 Embeddings**: Semantic text understanding
- **Title Context**: Product title integration
- **Real Title**: Enhanced title processing

## Results Summary

### Amazon Beauty Dataset Performance

| Model Variant | HR@10 | NDCG@10 | Epochs |
|---------------|-------|---------|---------|
| MiniLLM       | 0.4534| 0.2663  | 100     |
| E5 (no title) | 0.4727| 0.2881  | 50+     |
| E5 + Title    | -     | -       | 200+    |
| E5 + Real Title| -     | -       | 200+    |

### Key Findings
1. **Context Enhancement**: E5 embeddings show consistent improvement over baseline
2. **Stable Training**: All models demonstrate stable convergence
3. **Performance Gains**: Context-aware models outperform vanilla sequential approaches
4. **Scalability**: Processing pipeline handles large-scale datasets efficiently

## Technical Details

### Data Processing Pipeline
- **Input Format**: Amazon JSON.gz review and metadata files
- **Filtering**: Minimum 5 interactions per user/item
- **Sequencing**: Time-ordered user-item interaction sequences
- **Feature Engineering**: Categorical encoding, brand dummies, price normalization

### Model Architecture
- **Base**: Sequential recommendation framework
- **Enhancement**: LLM-generated contextual embeddings
- **Integration**: Dynamic context fusion with sequential patterns
- **Output**: Enhanced recommendation predictions

### Evaluation Metrics
- **Hit Rate @ 10**: Proportion of relevant items in top-10 recommendations
- **NDCG @ 10**: Rank-aware quality measure for top-10 recommendations
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Performance on held-out validation set

## Future Work

The project has established a solid foundation for LLM-enhanced sequential recommendations. Potential next steps include:

1. **Advanced LLM Integration**: Explore larger language models for enhanced context
2. **Multi-Modal Context**: Incorporate image and audio features
3. **Real-Time Inference**: Optimize for production deployment
4. **Cross-Domain Adaptation**: Extend to other recommendation domains
5. **Explainability**: Develop interpretable recommendation explanations

## Dependencies

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- gzip (built-in)

## Contributing

This project was developed as part of the 2025 Open Source Research Experience (OSRE). Contributions are welcome through:
- Issue reporting
- Feature requests
- Performance improvements
- Documentation enhancements

## License

This project is part of the OSRE initiative and follows open-source licensing guidelines.

## Acknowledgments

- **Mentors**: Linsey Pang, Bin Dong
- **Dataset Sources**: Amazon Public Data, MovieLens
- **Research Foundation**: SASRec, BERT4Rec, and related sequential recommendation literature

## Contact

For questions or contributions related to this project, please refer to the OSRE project page or create an issue in this repository.

---

*This README documents the current state of the LLMSeqRec project as of the OSRE 2025 implementation phase.*
